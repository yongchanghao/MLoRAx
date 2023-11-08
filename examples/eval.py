# Copyright (C) 2023 Yongchang Hao. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import math
import multiprocessing as mp
from pathlib import Path

import evaluate
import jax
import nltk
import numpy as np
import orbax.checkpoint as ocp
import tqdm
import transformers
from datasets import Dataset, load_dataset

from mlorax import LoRASpec, lora_init

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="t5-small")
parser.add_argument("--data", type=str, default="xsum")
parser.add_argument("--split", type=str, default="test")

parser.add_argument("--lora", type=str)

parser.add_argument("--rank", type=int, default=8)
parser.add_argument(
    "--rules",
    type=str,
    nargs="+",
    default=["Attention.q", "Attention.k", "Attention.v", "Attention.o"],
)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--tune-vectors", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lora-disabled", action="store_true")
parser.add_argument("--max-source-length", type=int, default=512)
parser.add_argument("--max-target-length", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=8)


metric = evaluate.load("rouge")


def shift_tokens_right(
    input_ids: np.array, pad_token_id: int, decoder_start_token_id: int
) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(
        shifted_input_ids == -100, pad_token_id, shifted_input_ids
    )
    return shifted_input_ids


def preprocess_function(examples):
    inputs = examples["document"]
    targets = examples["summary"]
    inputs = ["summarize: " + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target=targets,
        max_length=args.max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    model_inputs["labels"] = labels["input_ids"]
    decoder_input_ids = shift_tokens_right(
        labels["input_ids"],
        config.pad_token_id,
        config.decoder_start_token_id,
    )
    model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

    # We need decoder_attention_mask so we can ignore pad tokens from loss
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]

    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels
    )

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


args = parser.parse_args()
ckptr = ocp.PyTreeCheckpointer()

model = transformers.FlaxAutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
dataset = load_dataset(args.data, split=args.split)
config = transformers.AutoConfig.from_pretrained(args.model)

# dataset = dataset["test"]
dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=mp.cpu_count(),
    remove_columns=dataset.column_names,
    desc="Running tokenizer on prediction dataset",
)

if args.lora is not None and not args.lora_disabled:
    lora_spec = LoRASpec(
        rank=args.rank,
        rules=args.rules,
        alpha=args.alpha,
        tune_vectors=args.tune_vectors,
        seed=args.seed,
        disabled=args.lora_disabled,
    )
    trainable, _, merge_fn = lora_init(lora_spec, model)
    trainable = jax.tree_map(
        lambda x, y: x.at[:].set(y),
        trainable,
        ckptr.restore(Path(args.lora).absolute()),
    )
    params = merge_fn(trainable)

else:
    params = model.params


@jax.jit
def generate_step(params, batch):
    # Define generation function
    max_length = (
        args.max_target_length
        if args.max_target_length is not None
        else model.config.max_length
    )
    num_beams = model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def _single_generate(_minibatch):
        return model.generate(
            input_ids=_minibatch["input_ids"][None, ...],
            attention_mask=_minibatch["attention_mask"][None, ...],
            params=params,
            **gen_kwargs,
        ).sequences[0]

    return jax.vmap(_single_generate)(batch)


def data_loader(
    rng: jax.random.PRNGKey,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last=True,
):
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[
            : steps_per_epoch * batch_size
        ]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


batch_size = args.batch_size
eval_preds = []
eval_labels = []
loader = data_loader(
    jax.random.PRNGKey(0), dataset, batch_size, drop_last=False
)
steps = math.ceil(len(dataset) / batch_size)
for _ in tqdm.tqdm(
    range(steps),
    desc="Evaluating...",
    position=2,
    leave=False,
):
    # Model forward
    batch = next(loader)
    labels = batch["labels"]

    # generation
    generated_ids = generate_step(params, batch)
    eval_preds.extend(jax.device_get(generated_ids))
    eval_labels.extend(labels)

rouge_metrics = compute_metrics(eval_preds, eval_labels)
print(rouge_metrics)

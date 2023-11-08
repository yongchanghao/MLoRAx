# Adapted from HuggingFace's run_summarization_flax.py
# Modifications Copyright (C) 2023 Yongchang Hao. All rights reserved.
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

import logging
import math
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path
from typing import Callable

import datasets
import evaluate
import hydra
import jax
import jax.numpy as jnp
import nltk
import numpy as np
import optax
import orbax.checkpoint as ocp
import transformers
from datasets import Dataset, load_dataset
from filelock import FileLock
from flax import traverse_util
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.common_utils import onehot
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    set_seed,
)

from mlorax import LoRASpec, lora_init

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
datasets.disable_caching()


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray


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


def write_train_metric(summary_writer, train_metrics, train_time, step):
    length = len(train_metrics)
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train/{key}"
            summary_writer.scalar(tag, val, step - length + i + 1)
    summary_writer.scalar("train/time", train_time, step)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        tag = f"eval/{metric_name}"
        summary_writer.scalar(tag, value, step)


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=num_warmup_steps,
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def main(args):
    # Make one log on every process with the configuration for debugging.

    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # logger.info(f"Training/evaluation parameters {args.training}")

    # Set seed before initializing model.
    set_seed(args.training.seed)

    dataset = load_dataset(
        args.data.dataset_name,
        keep_in_memory=False,
    )

    # Load pretrained model and tokenizer

    config = AutoConfig.from_pretrained(
        args.model.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.model_name_or_path,
    )

    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
        args.model.model_name_or_path,
        config=config,
        seed=args.training.seed,
        dtype=getattr(jnp, args.model.dtype),
        from_pt=False,
    )

    rng = jax.random.PRNGKey(args.training.seed)
    rng, dropout_rng = jax.random.split(rng)

    prefix = "summarize: "
    # Preprocessing the datasets.

    # Get the column names for input/target.
    column_names = dataset["train"].column_names
    dataset_columns = summarization_name_mapping.get(
        args.data.dataset_name, None
    )
    text_column, summary_column = dataset_columns

    max_target_length = args.data.max_target_length

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

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=args.data.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
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

    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = dataset["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=mp.cpu_count(),
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

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

    # Enable tensorboard only on the master node

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
            log_dir=Path(args.training.output_dir)
        )
        summary_writer.hparams(vars(args.training))

    # Store some constant
    num_epochs = int(args.training.num_train_epochs)
    train_batch_size = (
        int(args.training.per_device_batch_size) * jax.device_count()
    )
    per_device_batch_size = int(args.training.per_device_batch_size)
    eval_batch_size = per_device_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    ckptr = ocp.PyTreeCheckpointer()

    # Create learning rate schedule

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {
            path: (
                path[-1] != "bias" and path[-2:] not in layer_norm_named_params
            )
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    optimizer_args = OmegaConf.to_object(args.optimizer)
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        args.training.num_train_epochs,
        args.training.warmup_steps,
        optimizer_args.pop("learning_rate"),
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.training.max_grad_norm),
        optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            **optimizer_args,
            mask=decay_mask_fn,
        ),
    )

    num_original_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y.size, model.params, 0
    )

    rng = jax.random.PRNGKey(args.training.seed)

    lora_spec = LoRASpec(
        rank=args.lora.rank,
        rules=args.lora.rules,
        alpha=args.lora.alpha,
        tune_vectors=args.lora.tune_vectors,
        seed=args.training.seed,
        disabled=args.lora.disabled,
        dropout=args.lora.dropout,
    )
    trainable, apply_fn, merge_fn = lora_init(lora_spec, model)
    state = TrainState.create(
        apply_fn=apply_fn,
        params=trainable,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    num_trainable_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y.size, trainable, 0
    )
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    logger.info(f"Number of original parameters: {num_original_params}")
    if jax.process_index() == 0:
        summary_writer.text(
            "num_trainable_params", str(num_trainable_params), 0
        )
        summary_writer.text("num_original_params", str(num_original_params), 0)

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence)
            + (vocab_size - 1)
            * low_confidence
            * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(
            labels, vocab_size, on_value=confidence, off_value=low_confidence
        )

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss / num_labels, num_labels

    dummy_batch = next(
        data_loader(rng, train_dataset, train_batch_size, shuffle=False)
    )

    # Define gradient update step fn
    @partial(
        jax.jit,
        donate_argnums=(0,),
    )
    def train_step(state, batch):
        # batch.shape = (dev * ga * bsz, seqlen)

        def compute_loss(params, inputs, rng):
            labels = inputs.pop("labels")
            logits = state.apply_fn(
                **inputs, params=params, dropout_rng=rng, train=True
            )[0]
            loss, num_labels = loss_fn(
                logits,
                labels,
                batch["decoder_attention_mask"],
                args.training.label_smoothing_factor,
            )
            return loss, num_labels

        (loss, num_labels), grad = jax.value_and_grad(
            compute_loss, has_aux=True
        )(state.params, batch, state.dropout_rng)
        new_dropout_rng = jax.random.split(state.dropout_rng)[-1]

        state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grad),
        }

        return state, metrics

    # Define eval fn
    @jax.jit
    def eval_step(state, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        loss, num_labels = loss_fn(
            logits,
            labels,
            batch["decoder_attention_mask"],
            label_smoothing_factor,
        )

        metrics = {"loss": loss}
        return metrics

    @jax.jit
    def generate_step(state, batch):
        # Define generation function
        max_length = (
            args.data.max_target_length
            if args.data.max_target_length is not None
            else model.config.max_length
        )
        num_beams = model.config.num_beams
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        output_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=merge_fn(state.params),
            **gen_kwargs,
        )
        return output_ids.sequences

    # dummy run

    flops = (
        train_step.lower(state, dummy_batch)
        .compile()
        .cost_analysis()[-1]["flops"]
    )
    state, _ = train_step(state, dummy_batch)
    memory = (
        sum(
            jax.devices()[i].memory_stats()["peak_bytes_in_use"]
            for i in range(jax.device_count())
        )
        / 1024**3
    )
    logger.info(f"Flops: {flops / (1024**3)} GF")
    logger.info(f"Peak memory usage: {memory} GiB")
    if jax.process_index() == 0:
        summary_writer.text("Flops (G)", str(flops / (1024**3)), 0)
        summary_writer.text("Memory (GiB)", str(memory), 0)

    # Replicate the train state on each device

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.training.per_device_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_start = time.time()
    train_metrics = []

    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_loader = data_loader(
            input_rng, train_dataset, train_batch_size, shuffle=True
        )
        # train
        for step in tqdm(
            range(steps_per_epoch), desc="Training...", position=1, leave=False
        ):
            batch = next(train_loader)  # (#dev * bsz, seqlen)
            state, train_metric = train_step(state, batch)
            train_metric = jax.device_get(train_metric)

            train_metrics.append(train_metric)

            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            if cur_step % args.training.logging_steps == 0 and cur_step > 0:
                train_time = time.time() - train_start
                if jax.process_index() == 0:
                    write_train_metric(
                        summary_writer, train_metrics, train_time, cur_step
                    )
                    train_metrics = []

            if (
                cur_step % args.training.eval_steps == 0
                or cur_step + 1 == total_train_steps
            ) and cur_step > 0:
                # ======================== Evaluating ==============================
                eval_metrics = []
                eval_preds = []
                eval_labels = []
                eval_loader = data_loader(
                    input_rng, eval_dataset, eval_batch_size, drop_last=False
                )
                eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
                for _ in tqdm(
                    range(eval_steps),
                    desc="Evaluating...",
                    position=2,
                    leave=False,
                ):
                    # Model forward
                    batch = next(eval_loader)
                    labels = batch["labels"]

                    metrics = eval_step(
                        state, batch, args.training.label_smoothing_factor
                    )
                    eval_metrics.append(metrics)

                    # generation
                    generated_ids = generate_step(state, batch)
                    eval_preds.extend(jax.device_get(generated_ids))
                    eval_labels.extend(labels)

                eval_keys = eval_metrics[0].keys()
                eval_metrics = {
                    k: jnp.stack(
                        [metrics[k] for metrics in eval_metrics]
                    ).mean()
                    for k in eval_keys
                }

                # compute ROUGE metrics
                rouge_metrics = compute_metrics(eval_preds, eval_labels)
                eval_metrics.update(rouge_metrics)

                # Print metrics and update progress bar
                desc = (
                    f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']}"
                )
                epochs.write(desc)
                epochs.desc = desc

                # Save model and metrics
                if jax.process_index() == 0:
                    ckptr.save(
                        Path(args.training.output_dir).absolute()
                        / f"ckpt_{cur_step}",
                        jax.device_get(state.params),
                    )
                    write_eval_metric(summary_writer, eval_metrics, cur_step)


@hydra.main(version_base=None, config_path=".", config_name="config")
def launch(args: OmegaConf) -> None:
    jax.config.update("jax_threefry_partitionable", True)
    if args.training.output_dir is not None:
        args.training.output_dir = os.path.expanduser(args.training.output_dir)

    main(args)


if __name__ == "__main__":
    launch()

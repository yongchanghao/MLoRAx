# Fine-tune T5-Small on summarization tasks

## Install dependencies

Let's create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate
```

**On your own:** install `JAX` properly on your device. See the [official guide](https://github.com/google/jax#installation) for more details.

Install all the other dependencies when `JAX` is installed.

```bash
pip install -r requirements.txt
```

## Basic: fine-tune T5-Small on XSUM

In `config.yaml`, we can use the following configuration to specify the LoRA configuration:

```yaml
lora:
  rank: 8
  rules:
    - "Attention.o"
    - "Attention.q"
    - "Attention.v"
    - "Attention.k"
  alpha: null
  tune_vectors: false
  disabled: false
  dropout: 0.0
```

This is specifically for HuggingFace's T5 models. For other models, you may need to change the rules.

These configurations are going to be passed to the `LoRASpec` class.

In `train.py`, you can see the following code:

```diff
+ lora_spec = LoRASpec(
+     rank=args.lora.rank,
+     rules=args.lora.rules,
+     alpha=args.lora.alpha,
+     tune_vectors=args.lora.tune_vectors,
+     seed=args.training.seed,
+     disabled=args.lora.disabled,
+     dropout=args.lora.dropout,
+ )
+ trainable, apply_fn, merge_fn = lora_init(lora_spec, model)
  state = TrainState.create(
      apply_fn=apply_fn,
      params=trainable,
      tx=optimizer,
      dropout_rng=dropout_rng,
  )
```

which is all we need to add to enable LoRA training.

To test your saved LoRA weights, run

```bash
python eval.py --lora <path_to_your_weights>
```

## Advanced: saving and loading checkpoints

To save the LoRA weights, we simply need to save the `trainable` variable. To load the LoRA weights, we need to load the `trainable` variable and then call `apply_fn` to get the full model weights.

In `train.py`, we save the weights in the following way, which is essentially the same as the original code.

```python
ckptr.save(
    Path(args.training.output_dir).absolute()
    / f"ckpt_{cur_step}",
    jax.device_get(state.params),
)
```

In `eval.py`, we load the weights in the following way:

```diff
- params = model.params
+ trainable, _, merge_fn = lora_init(lora_spec, model)
+ trainable = jax.tree_map(
+     lambda x, y: x.at[:].set(y),
+     trainable,
+     ckptr.restore(Path(args.lora).absolute()),
+ )
+ params = merge_fn(trainable)
```

The weights will be correctly loaded when `lora_spec` has the arguments as the one when saving the weights.

## Results on XSUM

THe following commands are used to reproduce the results on XSUM.

```bash
python train.py lora.rank=8  optimizer.learning_rate=1e-3  # LoRA-FT (8)
python train.py lora.rank=32  optimizer.learning_rate=1e-3  # LoRA-FT (32)
python train.py lora.disabled=true optimizer.learning_rate=1e-4  # Full-FT
```

You should get the results similar to the following:

|                    | Zero-Shot | LoRA-FT (8) | LoRA-FT (32) | Full-FT |
| ------------------ | --------- | ----------- | ------------ | ------- |
| # Trainable Params | -         | ~ 0.59M     | ~ 2.4M       | ~61M    |
| Compute (GFLOPs)\* | -         | 2.450       | 2.158        | 3.237   |
| VRAM (GB )\*       | -         | 1.317       | 1.350        | 1.85    |
| ROUGE-1            | 19.29     | 30.79       | 31.39        | 32.59   |
| ROUGE-2            | 2.84      | 8.87        | 9.35         | 10.35   |
| ROUGE-L            | 13.63     | 23.86       | 24.44        | 25.51   |

\* The FLOPs and VRAM are measured on a RTX Titan with CUDA 11 and `jax==0.4.20`.

## Conclusion

You can see that LoRA-FT significantly boosts the performance on the summarization task compared with the untuned T5 (Zero-Shot). It also reduces the number of trainable parameters and the compute cost compared with full fine-tuning, although the ROUGE scores are slightly lower. The performance can be improved by increasing the rank at a small cost of compute and memory.

## Installation

### Choice 1: Install with pip

You can install `mlorax` with pip. This is the recommended way to use `mlorax` if you want to receive future updates.

```bash
pip install mlorax
```

### Choice 2: Just copy the code

You can also directly copy the code from `mlorax.py` and paste it into your project. This is the easiest way to use `mlorax` if you do not care about future updates.

### Choice 3: Install from source

You can also install `mlorax` from source. You only need to do this if you want to contribute to the project.

```bash
git clone https://github.com/yongchanghao/MLoRAx.git
cd MLoRAx
pip install -e .
```

## Usage

It is extremely easy to use `mlorax` to convert any Flax model to a LoRA model. The following code snippet shows how to convert a T5 model to a LoRA model based on HuggingFace's [FlaxT5ForConditionalGeneration](https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5ForConditionalGeneration) class.

```diff
+ import mlorax
  model = FlaxT5ForConditionalGeneration.from_pretrained('t5-small')
- params = model.params
- apply_fn = model.__call__
+ lora_spec = mlorax.LoRASpec(rank=16, rules=['Attention.q', 'Attention.v'])
+ params, apply_fn, merge_fn = mlorax.lora_init(lora_spec, model)
  state = TrainState(apply_fn=apply_fn, params=params, tx=tx, **kwargs)
```

That's it! You can now train the model as usual.

### Principles

Always use the **returned** `apply_fn` for model forwarding if possible. Otherwise use `params=merge_fn(params)` to pass the merged parameters to the function. For example, if you want to use `model.generate` for text generation, you can do the following:

```diff
- outputs = model.generate(**batch, params=params)
+ outputs = model.generate(**batch, params=merge_fn(params))
```

## Example and Results

Please refer to the [examples](https://github.com/yongchanghao/MLoRAx/tree/main/examples) folder for details.

## Citation

If you find MLoRAx useful, please cite the following paper:

```bibtex
@software{hao2023lmrax,
  author = {Yongchang Hao},
  title = {{T}he {LMR}ax {E}cosystem: a minimalist library for training {T}ransformer models with {JAX}},
  year = {2023},
  url = {https://github.com/yongchanghao/LMRax},
  version = {0.9.5}
}
```

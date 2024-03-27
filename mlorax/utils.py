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


from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional
import logging
import inspect

import chex
import flax
import jax
import jax.numpy as jnp

PATH_SEP = "."
LORA_SIGNATURE = PATH_SEP + "$lora$"
LORA_A_SUFFIX = LORA_SIGNATURE + PATH_SEP + "a"
LORA_B_SUFFIX = LORA_SIGNATURE + PATH_SEP + "b"

RNG_KEYWORDS = [
    "dropout_rng",
    "dropout_prng",
    "dropout_key",
    "rng",
    "prng",
    "key",
]


class WeightState(Enum):
    FREEZED = 0
    FULL = 1
    FACTORIZED = 2


@dataclass
class LoRASpec:
    rank: int
    rules: Iterable[str]
    alpha: Optional[float] = None  # default to rank
    dropout: float = 0.0
    tune_vectors: bool = False
    tune_others: bool = False
    seed: int = 0
    disabled: bool = False
    logger_level: str = "INFO"


def _decision_fn(
    lora_spec: LoRASpec,
    name: str,
    param: chex.ArrayTree,
):
    """
    Decision function to determine the weight state of a parameter.
    """

    default = WeightState.FULL if lora_spec.tune_others else WeightState.FREEZED

    # check if the parameter is floating
    if not jnp.issubdtype(param.dtype, jnp.floating):
        return WeightState.FREEZED

    # check if the parameter is a high-rank tensor
    if param.ndim > 2:
        return default

    # check if the parameter is a vector
    if lora_spec.tune_vectors and param.ndim <= 1:
        return WeightState.FULL

    # factorize the parameter if it matches the rules
    for rule in lora_spec.rules:
        if rule in name:
            return WeightState.FACTORIZED

    # otherwise, return the default
    return default


def _lora_merge(
    lora_spec: LoRASpec,
    trainable: flax.core.FrozenDict,
    freezed: flax.core.FrozenDict,
    rng: Optional[chex.PRNGKey] = None,
):
    """
    Merge trainable and freezed parameters into a full parameter set.
    """
    trainable = flax.traverse_util.flatten_dict(trainable, sep=PATH_SEP)
    full_params = {}
    full_params.update(freezed)
    alpha = lora_spec.alpha if lora_spec.alpha is not None else lora_spec.rank

    trainable_paths = set()
    for path in trainable:
        if LORA_SIGNATURE not in path:
            # WeightState.FULL
            full_params[path] = trainable[path]
        else:
            # WeightState.FACTORIZED
            trainable_paths.add(path.split(LORA_SIGNATURE)[0])

    use_dropout = lora_spec.dropout > 0.0 and rng is not None
    keep_prob = 1 - lora_spec.dropout

    for path in trainable_paths:
        a = trainable[f"{path}{LORA_A_SUFFIX}"]
        if use_dropout:
            # math:
            #   dot(x * mask / (1-p), a)
            # = dot(x, diag(mask), a) / (1-p)
            # = dot(x, mask' * a / (1-p)), where mask'.shape = (n, 1)
            rng = jax.random.split(rng)[0]
            mask = jax.random.bernoulli(rng, p=keep_prob, shape=(a.shape[0], 1))
            a = mask * a / keep_prob
        b = trainable[f"{path}{LORA_B_SUFFIX}"]
        rank = a.shape[1]
        full_params[path] = jnp.matmul(a, b) * alpha / rank + freezed[path]

    return flax.traverse_util.unflatten_dict(full_params, sep=PATH_SEP)


def lora_init(
    lora_spec: LoRASpec,
    model: Any,
    params: Optional[flax.core.FrozenDict] = None,
    apply_fn: Optional[Callable] = None,
):
    """
    Initialize a model with LoRA parameters.
    Return a tuple of (trainable_params, apply_fn, merge_fn),
    where apply_fn: (trainable_params, *args, **kwargs) -> model_output
    and merge_fn: (trainable_params) -> full_params after merging.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(lora_spec.logger_level))

    if params is None:
        params = model.params
    if apply_fn is None:
        apply_fn = model.__call__

    if lora_spec.disabled:
        logger.info("LoRA is disabled.")
        trainable = params
        freezed = {}
    else:
        trainable = {}
        freezed = {}
        logger.info("Initializing LoRA...")
        rank = lora_spec.rank
        init_rng = jax.random.PRNGKey(lora_spec.seed)

        for path, weight in flax.traverse_util.flatten_dict(params, sep=PATH_SEP).items():
            weight_state = _decision_fn(lora_spec, path, weight)
            if weight_state == WeightState.FULL:
                logger.debug(f"Full: {path}({weight.dtype})={weight.shape}")
                trainable[path] = weight
            elif weight_state == WeightState.FREEZED:
                logger.debug(f"Freezed: {path}({weight.dtype})={weight.shape}")
                freezed[path] = weight
            elif weight_state == WeightState.FACTORIZED:
                logger.debug(f"Factorized: {path}({weight.dtype})={weight.shape}")
                trainable[f"{path}{LORA_A_SUFFIX}"] = jax.random.normal(
                    init_rng, (weight.shape[0], rank), dtype=weight.dtype
                ) / jnp.sqrt(weight.shape[0] / 2)
                trainable[f"{path}{LORA_B_SUFFIX}"] = jnp.zeros((rank, weight.shape[1]), dtype=weight.dtype)
                freezed[path] = weight
                init_rng = jax.random.split(init_rng)[0]
            else:
                raise ValueError(f"Unknown weight state: {weight_state}")

        trainable = flax.traverse_util.unflatten_dict(trainable, sep=PATH_SEP)

    def wrapped_apply_fn(
        params,
        lora_rng=None,
        lora_rng_detection=True,
        *args,
        **kwargs,
    ):
        """
        Apply the model with trainable parameters.
        Dropout is applied if
            - lora_rng is not None, or
            - kwargs[kw] is detected for kw in RNG_KEYWORDS
              when lora_rng_detection=True (default).
        """

        if "train" not in inspect.signature(apply_fn).parameters:
            if kwargs.get("train", None) is not None:
                kwargs["deterministic"] = not kwargs["train"]
                del kwargs["train"]

        if lora_rng is None and lora_rng_detection:
            for kw in RNG_KEYWORDS:
                if isinstance(kwargs.get(kw, None), chex.PRNGKey):
                    lora_rng = jax.random.split(kwargs[kw])[0]
                    break

        if lora_spec.disabled:
            return apply_fn(params=params, *args, **kwargs)

        return apply_fn(
            params=_lora_merge(
                lora_spec=lora_spec,
                trainable=params,
                freezed=freezed,
                rng=lora_rng,
            ),
            *args,
            **kwargs,
        )

    def wrapped_merge_fn(
        params,
        lora_rng=None,
    ):
        """
        Merge trainable and freezed parameters into a full parameter set.
        Dropout is applied if lora_rng is not None.
        """
        return _lora_merge(
            lora_spec=lora_spec,
            trainable=params,
            freezed=freezed,
            rng=lora_rng,
        )

    return trainable, wrapped_apply_fn, wrapped_merge_fn

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

import chex
import flax
import jax
import jax.numpy as jnp

PATH_SEP = "."
LORA_SIGNATURE = PATH_SEP + "$lora$"
LORA_A_SUFFIX = LORA_SIGNATURE + PATH_SEP + "a"
LORA_B_SUFFIX = LORA_SIGNATURE + PATH_SEP + "b"


class WeightState(Enum):
    FREEZED = 0
    FULL = 1
    FACTORIZED = 2


@dataclass
class LoRASpec:
    rank: int
    rules: Iterable[str]
    alpha: Optional[float] = None  # default to rank
    tune_vectors: bool = False
    seed: int = 0
    disabled: bool = False


def decision_fn(
    lora_spec: LoRASpec,
    name: str,
    param: chex.ArrayTree,
):
    """
    Decision function to determine the weight state of a parameter.
    """
    # check if the parameter is floating
    if not jnp.issubdtype(param.dtype, jnp.floating):
        return WeightState.FREEZED

    # check if the parameter is a high-rank tensor
    if param.ndim > 2:
        return WeightState.FREEZED

    # check if the parameter is a vector
    if lora_spec.tune_vectors and param.ndim <= 1:
        return WeightState.FULL

    # factorize the parameter if it matches the rules
    for rule in lora_spec.rules:
        if rule in name:
            return WeightState.FACTORIZED

    # otherwise, freeze the parameter
    return WeightState.FREEZED


def lora_merge(
    lora_spec: LoRASpec,
    trainable: flax.core.FrozenDict,
    freezed: flax.core.FrozenDict,
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

    for path in trainable_paths:
        a = trainable[f"{path}{LORA_A_SUFFIX}"]
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
    if params is None:
        params = model.params
    if apply_fn is None:
        apply_fn = model.__call__

    if lora_spec.disabled:
        return params, apply_fn, lambda params: params

    rank = lora_spec.rank
    rng = jax.random.PRNGKey(lora_spec.seed)

    trainable = {}
    freezed = {}
    for path, weight in flax.traverse_util.flatten_dict(
        params, sep=PATH_SEP
    ).items():
        weight_state = decision_fn(lora_spec, path, weight)
        if weight_state == WeightState.FULL:
            trainable[path] = weight
        elif weight_state == WeightState.FREEZED:
            freezed[path] = weight
        elif weight_state == WeightState.FACTORIZED:
            trainable[f"{path}{LORA_A_SUFFIX}"] = jax.random.normal(
                rng, (weight.shape[0], rank), dtype=weight.dtype
            ) / jnp.sqrt(weight.shape[0] / 2)
            trainable[f"{path}{LORA_B_SUFFIX}"] = jnp.zeros(
                (rank, weight.shape[1]), dtype=weight.dtype
            )
            freezed[path] = weight
            rng = jax.random.split(rng)[0]
        else:
            raise ValueError(f"Unknown weight state: {weight_state}")

    trainable = flax.traverse_util.unflatten_dict(trainable, sep=PATH_SEP)

    def wrapped_apply_fn(params, *args, **kwargs):
        return apply_fn(
            params=lora_merge(
                lora_spec=lora_spec, trainable=params, freezed=freezed
            ),
            *args,
            **kwargs,
        )

    def wrapped_merge_fn(params):
        return lora_merge(
            lora_spec=lora_spec, trainable=params, freezed=freezed
        )

    return trainable, wrapped_apply_fn, wrapped_merge_fn

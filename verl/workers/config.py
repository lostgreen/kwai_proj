# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
ActorRolloutRef config
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from .actor import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig
from .critic import CriticConfig
from .reward.config import RewardConfig
from .rollout import RolloutConfig


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "FSDPConfig",
    "ModelConfig",
    "OptimConfig",
    "RefConfig",
    "RewardConfig",
    "RolloutConfig",
    "WorkerConfig",
]


@dataclass
class WorkerConfig:
    hybrid_engine: bool = True
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.ref.teacher_models = self._normalize_ref_teacher_models()

        if self.ref.model.model_path is None and not self.ref.teacher_models:
            self.ref.model = deepcopy(self.actor.model)

        self.ref.micro_batch_size_per_device_for_experience = self.actor.micro_batch_size_per_device_for_experience
        self.ref.padding_free = self.actor.padding_free
        self.ref.dynamic_batching = self.actor.dynamic_batching
        self.ref.ulysses_size = self.actor.ulysses_size
        self.ref.use_torch_compile = self.actor.use_torch_compile

    def _normalize_ref_teacher_models(self) -> dict[str, ModelConfig]:
        teacher_models = {}
        for name, model_config in self.ref.teacher_models.items():
            if isinstance(model_config, ModelConfig):
                normalized = model_config
            elif isinstance(model_config, Mapping):
                normalized = ModelConfig(**model_config)
            else:
                normalized = ModelConfig(**dict(model_config))
            normalized.post_init()
            teacher_models[name] = normalized
        return teacher_models

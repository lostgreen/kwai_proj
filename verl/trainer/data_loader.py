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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.task_sampler import TaskHomogeneousBatchSampler
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_frames=config.max_frames,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    # ---- Task-homogeneous batching (每个 batch 内只含同一任务) ----
    if config.task_homogeneous_batching:
        task_weights = None
        if config.task_weights:
            # OmegaConf 已经把命令行 JSON/YAML 解析成 DictConfig，直接转成普通 dict
            task_weights = dict(config.task_weights)

        batch_sampler = TaskHomogeneousBatchSampler(
            dataset=train_dataset,
            batch_size=train_batch_size,
            task_key=config.task_key,
            task_weights=task_weights,
            seed=config.seed,
            drop_last=True,
        )
        train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
    else:
        # ---- 原始随机采样 ----
        if config.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(config.seed)
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

        train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_frames=config.max_frames,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader

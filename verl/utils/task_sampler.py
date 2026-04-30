# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
TaskHomogeneousBatchSampler: 保证每个 batch 内样本来自同一任务类型。

核心思路:
1. 数据集加载后，按 task_key 字段将样本索引分到不同 "桶"；
   可通过 task_grouping 将细粒度 problem_type 合并为粗粒度桶；
2. 每个桶内独立 shuffle；
3. 按 weighted round-robin 在桶之间轮替采样，每次取一个完整 batch；
4. 支持自定义各任务的采样权重（默认按数据量等比例）。

使用方式:
    sampler = TaskHomogeneousBatchSampler(
        dataset=train_dataset,
        batch_size=16,
        task_key="problem_type",
        task_grouping="raw",
        task_weights={"temporal_seg": 0.4, "add": 0.15, "delete": 0.15, "replace": 0.15, "sort": 0.15},
        seed=42,
        drop_last=True,
    )
    dataloader = StatefulDataLoader(dataset=train_dataset, batch_sampler=sampler, ...)
"""

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler

from .task_grouping import resolve_task_homogeneous_bucket


class TaskHomogeneousBatchSampler(Sampler[List[int]]):
    """
    每个 batch 只包含同一任务的样本。

    任务之间的采样顺序由 task_weights 控制（加权轮转）。
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        task_key: str = "problem_type",
        task_grouping: str = "raw",
        task_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
        drop_last: bool = True,
    ):
        """
        Args:
            dataset: 必须是 RLHFDataset 或类似对象，其 .dataset 有 __getitem__
            batch_size: 每个 batch 的样本数
            task_key: 用于区分任务的字段名
            task_grouping: task_key 值到 batch 桶的映射模式。raw 表示不合并；
                           opd_task_group 表示合并为 base/aot/seg/logic。
            task_weights: 各任务的采样权重（会自动归一化）。
                          None → 按数据量等比例
            seed: 随机种子
            drop_last: 不满一个 batch 的余量是否丢弃
        """
        self.batch_size = batch_size
        self.task_key = task_key
        self.task_grouping = task_grouping
        self.seed = seed
        self.drop_last = drop_last

        # 分桶
        self.task_buckets: Dict[str, List[int]] = defaultdict(list)
        hf_dataset = dataset.dataset  # HuggingFace Dataset 对象

        for idx in range(len(hf_dataset)):
            raw_task = hf_dataset[idx].get(task_key, "") or ""
            task = resolve_task_homogeneous_bucket(raw_task, task_grouping)
            self.task_buckets[task].append(idx)

        self.task_names = sorted(self.task_buckets.keys())
        print(f"[TaskHomogeneousBatchSampler] Found {len(self.task_names)} tasks: "
              f"{', '.join(f'{t}({len(self.task_buckets[t])})' for t in self.task_names)}")

        # 权重
        if task_weights is not None:
            self.task_weights = {}
            for t in self.task_names:
                self.task_weights[t] = task_weights.get(t, 1.0)
        else:
            # 按数据量等比
            self.task_weights = {t: len(self.task_buckets[t]) for t in self.task_names}

        # 归一化权重
        total_w = sum(self.task_weights.values())
        self.task_weights = {t: w / total_w for t, w in self.task_weights.items()}

        # 计算每个 epoch 各任务的 batch 数
        self._compute_task_batch_counts()

    def _compute_task_batch_counts(self):
        """根据权重计算每个 epoch 中各任务应产生的 batch 数。"""
        total_batches = 0
        for t in self.task_names:
            n = len(self.task_buckets[t])
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += math.ceil(n / self.batch_size)

        # 按权重分配
        self.task_batch_counts = {}
        for t in self.task_names:
            n = len(self.task_buckets[t])
            if self.drop_last:
                max_batches = n // self.batch_size
            else:
                max_batches = math.ceil(n / self.batch_size)
            # 按权重得到的理想 batch 数，但不超过该任务实际可产生的最大 batch 数
            ideal = int(total_batches * self.task_weights[t])
            self.task_batch_counts[t] = min(max(1, ideal), max_batches) if max_batches > 0 else 0

        self._total_batches = sum(self.task_batch_counts.values())
        print(f"[TaskHomogeneousBatchSampler] Batch allocation: "
              f"{', '.join(f'{t}={self.task_batch_counts[t]}' for t in self.task_names)} "
              f"(total={self._total_batches})")

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        self.seed += 1  # 每次迭代换种子，保证 epoch 间不重复

        # 每个桶独立 shuffle
        shuffled_buckets: Dict[str, List[int]] = {}
        for t in self.task_names:
            indices = list(self.task_buckets[t])
            rng.shuffle(indices)
            shuffled_buckets[t] = indices

        # 为每个桶切分 batches
        task_batches: Dict[str, List[List[int]]] = {}
        for t in self.task_names:
            indices = shuffled_buckets[t]
            batches = []
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    break
                batches.append(batch)
            # 只取 task_batch_counts[t] 个
            task_batches[t] = batches[: self.task_batch_counts[t]]

        # 加权交错排列：按照权重比例穿插不同任务的 batch
        # 策略：用 stride-interleave，让各任务 batch 均匀分布
        all_batches = self._interleave_batches(task_batches, rng)

        yield from all_batches

    def _interleave_batches(
        self,
        task_batches: Dict[str, List[List[int]]],
        rng: random.Random,
    ) -> List[List[int]]:
        """
        将各任务的 batch 均匀交错排列。

        使用 fractional stride 方法：
        - 为每个任务算出 stride = total / count
        - 在 [0, total) 上为每个 batch 分配一个"位置"
        - 按位置排序
        """
        total = sum(len(batches) for batches in task_batches.values())
        if total == 0:
            return []

        positioned: List[tuple] = []  # (position, batch)

        for t, batches in task_batches.items():
            n = len(batches)
            if n == 0:
                continue
            stride = total / n
            for i, batch in enumerate(batches):
                # 加小随机扰动避免同一位置冲突
                pos = i * stride + rng.uniform(0, stride * 0.3)
                positioned.append((pos, batch))

        positioned.sort(key=lambda x: x[0])
        return [batch for _, batch in positioned]

    def state_dict(self) -> dict:
        """支持 StatefulDataLoader 的 checkpoint resume。"""
        return {"seed": self.seed}

    def load_state_dict(self, state_dict: dict):
        """恢复采样器状态。"""
        self.seed = state_dict.get("seed", self.seed)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# -*- coding: utf-8 -*-
"""
时序分割 Reward 工具库 — IoU / NMS / Hungarian 匹配 / F1 计算。

提供基础组件供 hier_seg_reward.py / dp_f1_reward.py / mixed_proxy_reward.py 导入。
不再包含独立的 compute_score() 入口。
"""

import re
from typing import List, Optional, Tuple


# ===========================
# 常量
# ===========================
NMS_IOU_THR = 0.7       # NMS 合并阈值
FORMAT_BONUS = 0.05     # 格式正确但 IoU=0 时的基础奖励


# ===========================
# 正则
# ===========================
EVENTS_PATTERN = re.compile(
    r"<events>(.*?)</events>",
    re.DOTALL,
)
SEGMENT_PATTERN = re.compile(
    r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)


# ===========================
# 解析
# ===========================
def parse_segments(text: str, duration: Optional[float] = None) -> List[List[float]]:
    """从 <events>...</events> 块中提取有效 [start, end] 列表。"""
    if not text:
        return []
    text = str(text)
    events_match = EVENTS_PATTERN.search(text)
    if events_match is None:
        return []

    events_block = events_match.group(1)
    segments: List[List[float]] = []

    for m in SEGMENT_PATTERN.finditer(events_block):
        try:
            start = float(m.group(1))
            end = float(m.group(2))
        except (ValueError, TypeError):
            continue
        if start >= end or start < 0:
            continue
        if duration is not None and end > duration + 1e-6:
            continue
        segments.append([start, end])

    return segments


def has_events_tag(text: str) -> bool:
    """检查文本是否包含 <events>...</events> 标签"""
    return EVENTS_PATTERN.search(str(text)) is not None


# ===========================
# IoU
# ===========================
def temporal_iou(a: List[float], b: List[float]) -> float:
    """计算两个时间段的 1D IoU"""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    intersection = max(0.0, inter_end - inter_start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def ngiou(a: List[float], b: List[float]) -> float:
    """Normalized Generalized IoU for 1D intervals. Returns value in [0, 1].

    NGIoU = (GIoU + 1) / 2, where GIoU = IoU - (|C| - |A∪B|) / |C|.
    Key: even when pred and gt don't overlap, NGIoU > 0 (closer → higher).
    """
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    intersection = max(0.0, inter_end - inter_start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union <= 0:
        return 0.0
    iou = intersection / union

    c_len = max(a[1], b[1]) - min(a[0], b[0])
    if c_len <= 0:
        return 0.0

    giou = iou - (c_len - union) / c_len
    return (giou + 1.0) / 2.0


# ===========================
# NMS
# ===========================
def nms_1d(segments: List[List[float]], iou_thr: float = NMS_IOU_THR) -> List[List[float]]:
    """
    1D 非极大值抑制：按段长度降序排列，
    依次保留与已选段 IoU 均 < iou_thr 的段。
    """
    if len(segments) <= 1:
        return segments

    sorted_segs = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)
    kept: List[List[float]] = []

    for seg in sorted_segs:
        if all(temporal_iou(seg, k) < iou_thr for k in kept):
            kept.append(seg)

    kept.sort(key=lambda s: s[0])
    return kept


# ===========================
# 纯 Python 匈牙利匹配（Munkres 算法）
# ===========================
def _hungarian_assignment(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    纯 Python 匈牙利（Munkres）算法实现。
    输入: M×N cost matrix (M 行 N 列)
    输出: 最优匹配的 (row, col) 列表
    """
    n_rows = len(cost_matrix)
    if n_rows == 0:
        return []
    n_cols = len(cost_matrix[0])
    if n_cols == 0:
        return []

    # 使 matrix 为方阵（padding 零）
    n = max(n_rows, n_cols)
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n_cols):
            cost[i][j] = cost_matrix[i][j]

    # 行归约
    for i in range(n):
        min_val = min(cost[i])
        for j in range(n):
            cost[i][j] -= min_val

    # 列归约
    for j in range(n):
        min_val = min(cost[i][j] for i in range(n))
        for i in range(n):
            cost[i][j] -= min_val

    # 标记
    INF = float('inf')
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to col j
    way = [0] * (n + 1)

    # 重新用标准 Jonker-Volgenant 实现
    # 重置 cost matrix
    cost_jv = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n_rows):
        for j in range(n_cols):
            cost_jv[i + 1][j + 1] = cost_matrix[i][j]

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_jv[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # 提取匹配结果（过滤 padding）
    result = []
    for j in range(1, n + 1):
        if p[j] != 0 and p[j] <= n_rows and j <= n_cols:
            result.append((p[j] - 1, j - 1))

    return result


# ===========================
# F1-IoU 计算
# ===========================
def compute_f1_iou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
) -> float:
    """
    匈牙利匹配 F1-IoU:
    1. NMS 去重预测段
    2. 构建 IoU cost matrix → 匈牙利匹配
    3. recall    = Σ IoU_matched / N_gt
       precision = Σ IoU_matched / N_pred
       reward    = 2·R·P / (R+P)
    """
    # NMS 去重
    pred_segs = nms_1d(pred_segs)
    num_pred = len(pred_segs)
    num_gt = len(gt_segs)

    if num_pred == 0 or num_gt == 0:
        return 0.0

    # IoU cost matrix (num_pred × num_gt)
    cost_matrix = []
    for i in range(num_pred):
        row = []
        for j in range(num_gt):
            row.append(1.0 - temporal_iou(pred_segs[i], gt_segs[j]))
        cost_matrix.append(row)

    # 匈牙利匹配
    matches = _hungarian_assignment(cost_matrix)

    # F1-IoU
    matched_ious = [1.0 - cost_matrix[r][c] for r, c in matches]
    total_iou = sum(matched_ious)

    recall = total_iou / num_gt
    precision = total_iou / num_pred
    denom = recall + precision
    return float(2.0 * recall * precision / denom) if denom > 0.0 else 0.0


# ===========================
# F1-NGIoU 计算
# ===========================
def compute_f1_ngiou(
    pred_segs: List[List[float]],
    gt_segs: List[List[float]],
    margin: float = 0.0,
) -> float:
    """
    F1-NGIoU（无 NMS）:
    1. 可选 margin: GT 边界外扩 margin 秒（L1 宽容模式）
       对每个 (pred, gt) 对取 max(ngiou(pred, gt), ngiou(pred, expanded_gt))
       确保 margin 只会提高分数，不会降低
    2. 构建 NGIoU cost matrix → 匈牙利匹配
    3. recall    = Σ NGIoU_matched / N_gt
       precision = Σ NGIoU_matched / N_pred
       reward    = 2·R·P / (R+P)
    """
    num_pred = len(pred_segs)
    num_gt = len(gt_segs)

    if num_pred == 0 or num_gt == 0:
        return 0.0

    # NGIoU cost matrix (num_pred × num_gt)
    cost_matrix = []
    if margin > 0:
        expanded_gt = [[max(0.0, g[0] - margin), g[1] + margin] for g in gt_segs]
        for i in range(num_pred):
            row = []
            for j in range(num_gt):
                # Take max of original and expanded → margin only helps
                score = max(ngiou(pred_segs[i], gt_segs[j]),
                            ngiou(pred_segs[i], expanded_gt[j]))
                row.append(1.0 - score)
            cost_matrix.append(row)
    else:
        for i in range(num_pred):
            row = []
            for j in range(num_gt):
                row.append(1.0 - ngiou(pred_segs[i], gt_segs[j]))
            cost_matrix.append(row)

    # 匈牙利匹配
    matches = _hungarian_assignment(cost_matrix)

    # F1-NGIoU
    matched_scores = [1.0 - cost_matrix[r][c] for r, c in matches]
    total_score = sum(matched_scores)

    recall = total_score / num_gt
    precision = total_score / num_pred
    denom = recall + precision
    return float(2.0 * recall * precision / denom) if denom > 0.0 else 0.0

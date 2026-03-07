# -*- coding: utf-8 -*-
"""
YouCook2 视频时序分割 Reward 函数 — F1-IoU + NMS + 格式奖励。

参考自 youcook_proxy/reward.py（ms-swift 版本），适配 EasyR1 batch reward 接口。

Reward = max(FORMAT_BONUS, matched_f1)

  1. NMS 去重：合并 IoU > NMS_IOU_THR 的重叠预测段
  2. 匈牙利匹配：一对一二分图匹配
  3. F1-IoU = 2·R·P / (R+P)
     - recall    = Σ IoU_matched / N_gt
     - precision = Σ IoU_matched / N_pred (NMS 后)
  4. 格式奖励：输出合法 <events>[...]</events> 给 FORMAT_BONUS 基础分

注意：不依赖 scipy，使用纯 Python 匈牙利匹配实现。
"""

import re
import random
from typing import Any, Dict, List, Optional, Tuple


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
# 主接口 (EasyR1 batch reward)
# ===========================
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    Batch reward 接口（与 EasyR1 兼容）。

    返回 list of dict:
        - overall: 综合分数 = 只看 f1_iou
        - format: 这里不再给固定得保底分，只在严苛正确时有微小起步分(其实可以干脆置 0，全靠 accuracy)
        - accuracy: F1-IoU 分数
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for idx, item in enumerate(reward_inputs):
        try:
            raw_response = item.get("response", "") or ""
            raw_gt = item.get("ground_truth", "") or ""

            # 解析 GT segments
            gt_segs = parse_segments(raw_gt)
            if not gt_segs:
                results.append({
                    "overall": 0.0, "format": 0.0,
                    "accuracy": 0.0, "structure_reward": 0.0,
                })
                continue
            
            # --- 严格反黑客过滤 (Anti-Reward Hacking) ---
            # 如果出现类似 "[数字-数字]" 的破折号畸形
            if re.search(r"\[\d+-\d+\]", raw_response):
                results.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0, "structure_reward": 0.0})
                continue
                
            # 反多重标签复读 (比如模型疯狂重复 </events> 或者 <events>)
            if raw_response.count("</events>") > 1 or raw_response.count("<events>") > 1:
                results.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0, "structure_reward": 0.0})
                continue
                
            # 格式检查
            has_format = has_events_tag(raw_response)
            
            if not has_format:
                results.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0, "structure_reward": 0.0})
                continue

            # 解析模型预测
            pred_segs = parse_segments(raw_response)

            if not pred_segs:
                # 给定 format 要求下没有合法数字也直接0分，不再施舍 BONUS
                results.append({
                    "overall": 0.0, "format": 0.0,
                    "accuracy": 0.0, "structure_reward": 0.0,
                })
                continue

            # 计算 F1-IoU reward
            f1_reward = compute_f1_iou(pred_segs, gt_segs)
            
            # 彻底取消兜底，让 accuracy 和 overall 完全对齐
            overall = f1_reward
            format_score = 0.0 # 只有当模型真正有收益时才会体现出价值，不白给格式分

            results.append({
                "overall": float(overall),
                "format": float(format_score),
                "accuracy": float(f1_reward),
                "structure_reward": 0.0,
            })

        except Exception:
            results.append({
                "overall": 0.0, "format": 0.0,
                "accuracy": 0.0, "structure_reward": 0.0,
            })

    # 日志采样
    if random.random() < 0.05: # 加大采样概率方便观察
        for idx, item in enumerate(reward_inputs):
            gt_segs = parse_segments(item.get("ground_truth", ""))
            pred_segs = parse_segments(item.get("response", ""))
            print(f"[YC2-Reward] GT_Raw:\n{item.get('ground_truth', '')}\nResp_Raw:\n{item.get('response', '')}")
            print(f"[YC2-Reward] gt_segs={gt_segs}")
            print(f"[YC2-Reward] pred_segs={pred_segs}")
            print(f"[YC2-Reward] scores={results[idx]}")

    return results

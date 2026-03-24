# -*- coding: utf-8 -*-
"""
混合代理任务 + 时序分割统一 Reward 函数。

支持 9 种任务类型（由 problem_type 字段区分）:

1. add      — 选择题：选下一步视频，精确匹配字母
2. delete   — 选择题：找出不属于序列的视频，精确匹配字母
3. replace  — 选择题：选填缺失步骤的视频，精确匹配字母
4. sort     — 排序题：按时间排列视频片段，jigsaw displacement reward
5. temporal_seg — 时序分割：F1-IoU reward（复用 youcook2_temporal_seg_reward）
6. aot_v2t      — 选择题：判断单段视频的时间方向 (A/B)
7. aot_t2v      — 选择题：在双段视频中匹配描述 (A/B)
8. aot_3way_v2t — 选择题：三选一视频-文本匹配 (A/B/C)
9. aot_3way_t2v — 选择题：三选一文本-视频匹配 (A/B/C)

格式要求（严格模式）:
- add/delete/replace: 必须包含 <answer>字母</answer>，
    缺少标签直接返回 0.0，不再回退到全文提取。
- sort: 必须包含 <answer>数字序列</answer>，
    缺少标签直接返回 0.0。
- temporal_seg: 答案为事件标签格式 (<events>...</events>)，无标签返回 0.0。

Reward 输出格式（兼容 EasyR1 batch reward 接口）:
    {
        "overall": float,    # 总分 (0.0 = 格式错误，0.0-1.0 = 准确率)
        "format":  float,    # 格式分 (1.0 = <answer> 标签存在，0.0 = 缺标签)
        "accuracy": float,   # 准确率奖励
    }
"""

import random
import re
from typing import Any, Dict, List, Optional

from verl.reward_function.temporal_grounding_reward import (
    temporal_grounding_reward,
)
from verl.reward_function.youcook2_temporal_seg_reward import (
    compute_f1_iou,
    has_events_tag,
    parse_segments,
)


# ===================================================================
# ① 选择题 Reward (add / delete / replace)
# ===================================================================

# 匹配 <answer> 标签内的内容
_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)

# 匹配单个大写字母（独立单词）
_SINGLE_LETTER_PATTERN = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def _extract_from_answer_tag(response: str) -> Optional[str]:
    """
    从 <answer>...</answer> 标签中提取原始文本。
    如果有多个标签，取最后一个。
    """
    matches = _ANSWER_TAG_PATTERN.findall(response)
    if matches:
        return matches[-1].strip()
    return None


def _has_required_tags(response: str) -> bool:
    """
    严格格式检查：必须包含 <answer> 标签。
    缺少视为格式错误，返回 False。
    """
    return _ANSWER_TAG_PATTERN.search(response) is not None


def _extract_choice(response: str) -> Optional[str]:
    """
    严格模式：仅从 <answer> 标签内提取选项字母，不做全文 fallback。
    调用前应已通过 _has_required_tags() 检查。
    """
    tag_content = _extract_from_answer_tag(response)
    if tag_content is None:
        return None
    letters = _SINGLE_LETTER_PATTERN.findall(tag_content)
    if not letters:
        return None
    return letters[-1].upper()


def _choice_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    选择题精确匹配 reward（严格格式模式）。

    必须包含 <answer> 标签，缺失直接返回 0.0。

    - 格式正确 + 答案正确: overall=1.0, format=1.0, accuracy=1.0
    - 格式正确 + 答案错误: overall=0.0, format=1.0, accuracy=0.0
    - 格式错误（缺 <answer>）: overall=0.0, format=0.0, accuracy=0.0
    """
    # 严格格式门控：缺少 <answer> 直接返回 0
    if not _has_required_tags(response):
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    gt = ground_truth.strip().upper()
    pred = _extract_choice(response)

    if pred is None:
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}

    accuracy = 1.0 if pred == gt else 0.0
    return {"overall": float(accuracy), "format": 1.0, "accuracy": float(accuracy)}


# ===================================================================
# ② 排序题 Reward (sort) — Temporal Jigsaw
# ===================================================================

# 匹配单个数字或空格分隔的数字
_DIGIT_PATTERN = re.compile(r"\d")


def _parse_sort_digits(text: str) -> Optional[List[int]]:
    """
    解析排序序列数字版本 "13245" 或 "1 3 2 4 5" → [1, 3, 2, 4, 5]。

    新格式：数字序列，1-索引，代表视频标号顺序。
    支持：
    - 连续数字: "13245" → [1, 3, 2, 4, 5]
    - 空格分隔: "1 3 2 4 5" → [1, 3, 2, 4, 5]
    - 混合: "1, 3, 2, 4, 5" → [1, 3, 2, 4, 5]
    """
    if not text:
        return None

    # 找所有单个数字
    digits = _DIGIT_PATTERN.findall(text)

    if not digits:
        return None

    # 转为整数
    try:
        int_seq = [int(d) for d in digits]
        if len(int_seq) < 2:
            return None
        return int_seq
    except (ValueError, TypeError):
        return None



def _compute_jigsaw_displacement(pred_seq: List[int], gt_seq: List[int]) -> float:
    """
    Temporal Jigsaw Reward (公式 7, 8)，针对整数序列版本:

    E_jigsaw = Σ |pos(k, P_hat) - pos(k, P_gt)|
    E_max    = Σ |i - (n-1-i)|  for i in 0..n-1  (reversed sequence)
    R_jigsaw = 1 - E_jigsaw / E_max

    返回 [0, 1] 的 reward。
    """
    n = len(gt_seq)
    if n <= 1:
        return 1.0 if pred_seq == gt_seq else 0.0

    # 建立 gt 元素 → 位置映射 (0-索引)
    gt_pos = {elem: i for i, elem in enumerate(gt_seq)}

    # 计算 E_jigsaw
    e_jigsaw = 0.0
    for i, elem in enumerate(pred_seq):
        gt_p = gt_pos.get(elem)
        if gt_p is None:
            # 预测中出现了 gt 中没有的数字，惩罚最大位移
            e_jigsaw += n - 1
        else:
            e_jigsaw += abs(i - gt_p)

    # E_max: 完全逆序的位移
    e_max = sum(abs(i - (n - 1 - i)) for i in range(n))
    if e_max == 0:
        return 1.0

    reward = 1.0 - e_jigsaw / e_max
    return max(0.0, reward)


def _sort_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    排序题 Reward（数字序列版本，严格格式模式）:
    - 必须包含 <answer> 标签，缺失直接返回 0.0
    - 仅从 <answer> 标签内解析数字序列，不做全文 fallback
    - 计算 jigsaw displacement reward
    """
    gt_seq = _parse_sort_digits(ground_truth)
    if gt_seq is None:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 严格格式门控
    if not _has_required_tags(response):
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    tag_content = _extract_from_answer_tag(response)
    pred_seq = _parse_sort_digits(tag_content) if tag_content is not None else None

    if pred_seq is None:
        return {"overall": 0.0, "format": 1.0, "accuracy": 0.0}

    # 长度不匹配的处理
    if len(pred_seq) != len(gt_seq):
        if len(pred_seq) > len(gt_seq):
            pred_seq = pred_seq[:len(gt_seq)]
        else:
            missing = [e for e in gt_seq if e not in pred_seq]
            pred_seq = pred_seq + missing[:len(gt_seq) - len(pred_seq)]

    jigsaw_r = _compute_jigsaw_displacement(pred_seq, gt_seq)
    accuracy = 1.0 if pred_seq == gt_seq else jigsaw_r
    return {"overall": float(accuracy), "format": 1.0, "accuracy": float(accuracy)}


# ===================================================================
# ③ 时序分割 Reward (temporal_seg) — 复用已有 F1-IoU 逻辑
# ===================================================================


def _temporal_seg_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    时序分割 Reward：F1-IoU。
    无格式奖励：无法解析 = 0.0。
    """
    gt_segs = parse_segments(ground_truth)
    if not gt_segs:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 反黑客：畸形格式
    if re.search(r"\[\d+-\d+\]", response):
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
    if response.count("</events>") > 1 or response.count("<events>") > 1:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    has_format = has_events_tag(response)
    if not has_format:
        # 无 <events> 标签 → 无奖励
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    pred_segs = parse_segments(response)
    if not pred_segs:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    f1_reward = compute_f1_iou(pred_segs, gt_segs)

    return {
        "overall": float(f1_reward),
        "format": 0.0,
        "accuracy": float(f1_reward),
    }



# ===================================================================
# ④ 统一 dispatch 入口 (EasyR1 batch reward 接口)
# ===================================================================

# 任务 → reward 函数映射
_TASK_REWARD_DISPATCH = {
    "add":          _choice_reward,
    "delete":       _choice_reward,
    "replace":      _choice_reward,
    "aot_v2t":      _choice_reward,
    "aot_t2v":      _choice_reward,
    "aot_3way_v2t": _choice_reward,
    "aot_3way_t2v": _choice_reward,
    "sort":         _sort_reward,
    "temporal_seg": _temporal_seg_reward,
    "temporal_grounding": None,  # 特殊处理：需要 metadata
}


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """
    统一多任务 Reward 计算入口。

    根据 reward_input["problem_type"] 自动分派到对应的 reward 函数。

    Args:
        reward_inputs: list of dict, 每个 dict 包含:
            - response: str (模型回复)
            - ground_truth: str (标准答案)
            - problem_type: str (任务类型: add/delete/replace/aot_v2t/aot_t2v/sort/temporal_seg)
            - data_type: str (数据类型, 通常为 "video")

    Returns:
        list of dict, 每个 dict 包含:
            - overall: float (总分)
            - format: float (格式分，总是 0.0)
            - accuracy: float (准确率分)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    results: List[Dict[str, float]] = []

    for idx, item in enumerate(reward_inputs):
        try:
            response = item.get("response", "") or ""
            ground_truth = item.get("ground_truth", "") or ""
            problem_type = item.get("problem_type", "") or ""

            # 查找对应的 reward 函数
            reward_fn = _TASK_REWARD_DISPATCH.get(problem_type)

            # temporal_grounding: 需要 metadata（含 duration）
            if problem_type == "temporal_grounding":
                metadata = item.get("metadata") or {}
                score = temporal_grounding_reward(response, ground_truth, metadata)
                results.append(score)
                continue

            if reward_fn is None:
                # 未知任务类型：尝试猜测
                # 如果答案看起来是单字母，用选择题；如果有 <events>，用分割；如果是数字序列，用排序
                if re.match(r"^\s*[A-Z]\s*$", ground_truth):
                    reward_fn = _choice_reward
                elif "<events>" in ground_truth:
                    reward_fn = _temporal_seg_reward
                elif re.match(r"^\d+$", ground_truth.strip()):
                    reward_fn = _sort_reward
                else:
                    # 默认精确匹配
                    reward_fn = _choice_reward

            score = reward_fn(response, ground_truth)
            results.append(score)

        except Exception:
            # 防止单样本异常影响整个 batch
            results.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0})

    # 采样日志（5% 概率打印）
    if random.random() < 0.05:
        _log_samples(reward_inputs, results)

    return results


def _log_samples(
    reward_inputs: List[Dict[str, Any]],
    results: List[Dict[str, float]],
    max_log: int = 5,
):
    """随机采样打印奖励计算结果，方便调试。"""
    indices = list(range(len(reward_inputs)))
    random.shuffle(indices)
    for idx in indices[:max_log]:
        item = reward_inputs[idx]
        score = results[idx]
        task = item.get("problem_type", "?")
        gt = item.get("ground_truth", "")[:120]
        resp = item.get("response", "")[:200]
        print(
            f"[MixedReward] task={task} | gt={gt!r} | "
            f"resp={resp!r} | scores={score}"
        )

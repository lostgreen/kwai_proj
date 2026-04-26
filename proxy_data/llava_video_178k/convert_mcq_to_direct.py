"""
将 LLaVA MCQ JSONL 转为 direct-answer 格式（无 CoT）。

关键变更：
  1. prompt 末尾指令改为: "Provide your answer (a single letter) inside <answer></answer> tags."
  2. 确保每条记录都有 `messages` 字段 — 这样 verl/utils/dataset.py 会直接使用
     messages 中的 prompt，跳过 QUESTION_TEMPLATE（避免被强制追加 CoT 指令）。

用法:
    python convert_mcq_to_direct.py input.jsonl output.jsonl
    python convert_mcq_to_direct.py input.jsonl  # 自动生成 input_direct.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

# ---- 需要匹配和替换的指令尾部 ----

# 1. 当前 prepare_mcq.py 的 direct 格式 (无 <answer> tag)
_OLD_DIRECT = "Answer with the option letter."

# 2. 可能存在的 CoT 格式
_COT_PATTERN = re.compile(
    r"(?:Think step by step inside <think>\s*</think> tags,?\s*then\s*)?"
    r"[Pp]rovide your (?:final )?answer.*?inside <answer>\s*</answer> tags\.?",
    re.DOTALL,
)

# 3. 另一种 CoT 格式
_COT_PATTERN_2 = re.compile(
    r"Think step by step inside <think>\s*</think> tags.*$",
    re.DOTALL,
)

# ---- 新的 direct-answer 指令 ----
DIRECT_INSTRUCTION = (
    "Provide your answer (a single letter) inside <answer></answer> tags."
)


def rewrite_prompt(prompt: str) -> tuple[str, bool]:
    """替换 prompt 末尾的指令为 direct-answer 格式。

    Returns:
        (new_prompt, changed)
    """
    # 尝试替换 CoT 模式
    new, n = _COT_PATTERN.subn(DIRECT_INSTRUCTION, prompt)
    if n > 0:
        return new.rstrip(), True

    new, n = _COT_PATTERN_2.subn(DIRECT_INSTRUCTION, prompt)
    if n > 0:
        return new.rstrip(), True

    # 替换旧的 direct 格式 (无 <answer> tag)
    if _OLD_DIRECT in prompt:
        new = prompt.replace(_OLD_DIRECT, DIRECT_INSTRUCTION)
        return new.rstrip(), True

    # 如果已经是目标格式，不改
    if DIRECT_INSTRUCTION in prompt:
        return prompt, False

    # fallback: 追加到末尾
    return prompt.rstrip() + "\n\n" + DIRECT_INSTRUCTION, True


def ensure_messages(record: dict, prompt: str) -> None:
    """确保 record 有 messages 字段，避免 dataset.py 走 QUESTION_TEMPLATE 路径。"""
    record["messages"] = [{"role": "user", "content": prompt}]


def process_file(input_path: str, output_path: str) -> None:
    converted = 0
    unchanged = 0
    total = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            prompt = record.get("prompt", "")

            new_prompt, changed = rewrite_prompt(prompt)
            record["prompt"] = new_prompt

            # 关键：写入 messages，让 dataset.py 走 messages 分支（跳过 QUESTION_TEMPLATE）
            ensure_messages(record, new_prompt)

            if changed:
                converted += 1
            else:
                unchanged += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {total} total, {converted} converted, {unchanged} unchanged")
    print(f"  -> {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl> [output.jsonl]")
        print(f"  If output is omitted, writes to <input>_direct.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_direct"))

    if input_path == output_path:
        # 原地覆盖: 先写临时文件再替换
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", dir=Path(input_path).parent,
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
        process_file(input_path, tmp_path)
        Path(tmp_path).replace(input_path)
        print(f"  (in-place overwrite: {input_path})")
    else:
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()

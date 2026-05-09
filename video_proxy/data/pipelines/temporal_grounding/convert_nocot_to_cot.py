"""
将 no_cot JSONL 转换为 CoT JSONL。
只替换 prompt 中的指令部分，其余字段不变。

用法:
    python convert_nocot_to_cot.py input.jsonl output.jsonl
"""

import json
import sys

# no_cot 独有的指令片段
NO_COT_INSTRUCTION = (
    'Output format (strictly follow this):\n'
    '<events>\n'
    '[start_time, end_time]\n'
    '</events>\n\n'
    'Where start_time and end_time are in seconds '
    '(precise to one decimal place, e.g., [12.5, 17.8]).'
)

# CoT 替换后的指令片段
COT_INSTRUCTION = (
    'First, think step by step inside <think></think> tags. '
    'Describe what happens at different time periods in the video '
    'and determine when the target event occurs.\n\n'
    'Then, provide the precise time period in the following format:\n'
    '<events>\n'
    '[start_time, end_time]\n'
    '</events>\n\n'
    'Where start_time and end_time are in seconds '
    '(precise to one decimal place, e.g., [12.5, 17.8]).'
)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_nocot.jsonl> <output_cot.jsonl>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            prompt = record.get("prompt", "")

            if NO_COT_INSTRUCTION in prompt:
                new_prompt = prompt.replace(NO_COT_INSTRUCTION, COT_INSTRUCTION)
                record["prompt"] = new_prompt
                # 同步更新 messages
                if record.get("messages"):
                    record["messages"][0]["content"] = new_prompt
                converted += 1
            else:
                skipped += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {converted} converted, {skipped} skipped -> {output_path}")


if __name__ == "__main__":
    main()

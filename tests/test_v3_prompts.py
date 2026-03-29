#!/usr/bin/env python3
"""验证 V3 prompt 模板的 {duration} 占位符能否正确格式化"""
import sys
sys.path.insert(0, "local_scripts/hier_seg_ablations/prompt_ablation")
from prompt_variants_v3 import PROMPT_VARIANTS_V3

errors = []
for level in ["L1", "L2", "L3"]:
    for variant in ["V1", "V2", "V3", "V4"]:
        tmpl = PROMPT_VARIANTS_V3[level][variant]
        try:
            result = tmpl.format(duration=128)
            assert "128" in result, f"{level}/{variant}: duration not found"
            assert "{duration}" not in result, f"{level}/{variant}: unformatted placeholder"
            print(f"  OK  {level}/{variant} ({len(result)} chars)")
        except Exception as e:
            errors.append(f"{level}/{variant}: {e}")
            print(f"  FAIL {level}/{variant}: {e}")

if errors:
    print(f"\n{len(errors)} errors!")
    sys.exit(1)
else:
    print("\nAll 12 templates format correctly!")

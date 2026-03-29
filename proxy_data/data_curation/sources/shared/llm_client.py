"""
Shared LLM client for two-stage data curation pipeline.

Handles:
- OpenAI-compatible API calls with retries
- JSON extraction from LLM responses (markdown fences, etc.)
- Thread-safe concurrent evaluation with progress tracking
- Resume (checkpoint) support
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def call_llm(
    messages: list[dict],
    api_base: str,
    api_key: str,
    model: str,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 400,
) -> dict:
    """Call OpenAI-compatible API and return parsed JSON dict."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    key = (
        api_key
        or os.environ.get("NOVITA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    client = OpenAI(api_key=key, base_url=api_base)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            return parse_json(content)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2**attempt)

    return {"error": str(last_error), "_parse_error": True}


def parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown wrapping."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try stripping markdown code fences
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{[\s\S]+\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return {"_raw_response": text, "_parse_error": True}


def load_checkpoint(output_path: str, id_field: str) -> tuple[set[str], list[dict]]:
    """Load previously assessed samples for resume support.

    Returns:
        (assessed_ids, existing_results)
    """
    assessed_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    assessed_ids.add(item.get(id_field, ""))
                    existing.append(item)
    return assessed_ids, existing


def run_concurrent_assessment(
    samples: list[dict],
    assess_fn,
    workers: int = 8,
    log_every: int = 20,
    score_field: str | None = None,
) -> tuple[list[dict], int]:
    """Run assess_fn concurrently on samples.

    Args:
        samples: list of samples to assess
        assess_fn: callable(sample) -> dict
        workers: number of concurrent threads
        log_every: log progress every N completions
        score_field: optional field path for progress logging (e.g. "l2_fit_score")

    Returns:
        (results, failed_count)
    """
    results: list[dict] = []
    failed = 0

    print(f"\n开始评估 {len(samples)} 条样本 (workers={workers})...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(assess_fn, s): i for i, s in enumerate(samples)
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                assessed = future.result()
                results.append(assessed)
                if score_field and (i + 1) % log_every == 0:
                    assessment = assessed.get("_assessment", {})
                    score = assessment.get(score_field, "?")
                    decision = assessment.get("decision", "?")
                    print(f"  [{i+1}/{len(samples)}] {score_field}={score} decision={decision}")
            except Exception as e:
                failed += 1
                print(f"  样本 {futures[future]} 失败: {e}")

    return results, failed


def write_results(
    results: list[dict],
    output_path: str,
    strip_fields: list[str] | None = None,
):
    """Write results to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            row = dict(r)
            for field in strip_fields or []:
                row.pop(field, None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  -> {output_path} ({len(results)} 条)")


def stratified_sample(
    samples: list[dict],
    n: int,
    domain_field: str = "source",
) -> list[dict]:
    """Stratified sampling by domain."""
    by_domain: dict[str, list] = {}
    for s in samples:
        d = s.get(domain_field, "unknown")
        by_domain.setdefault(d, []).append(s)

    import random
    per_domain = max(1, n // len(by_domain))
    chosen: list[dict] = []
    for domain, items in by_domain.items():
        k = min(per_domain, len(items))
        chosen.extend(random.sample(items, k))

    remaining = [s for s in samples if s not in chosen]
    if len(chosen) < n and remaining:
        chosen.extend(random.sample(remaining, min(n - len(chosen), len(remaining))))

    return chosen

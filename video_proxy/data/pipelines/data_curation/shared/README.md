# Shared Local Scoring

`local_screen.py` is the retained local VLM scorer. It is optional and is called
through `../curation/local_score.py` when `LOCAL_SCORE=1`.

The default curation path does not run a model; it only filters by duration and
writes `screen_keep.jsonl` for downstream compatibility.

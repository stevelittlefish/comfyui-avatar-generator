# TODO Checklist

A practical fix list from the code review. Keep it useful, keep it silly. Arrr.

## High priority

- [x] Validate CLI numeric inputs in `avatar_gen.py`
  - [x] Reject `--count < 1`
  - [x] Reject `--images-per < 1`
  - Suggested location: after `args = parser.parse_args()`
  - Suggested behavior:
    ```py
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.images_per < 1:
        parser.error("--images-per must be >= 1")
    ```

- [x] Fix documentation/default drift
  - [x] `avatar_gen.py` currently defaults to `--count 1 --images-per 1`
  - [x] `README.md` and/or `AGENTS.md` mention older defaults like `--count 20 --images-per 2`
  - [x] Decide whether to update docs or restore old defaults
  - [x] Make README usage examples match actual behavior

## Medium priority

- [x] Make `sort_avatars.py` category parsing stricter
  - Current code checks substring membership:
    ```py
    for cat in CATEGORIES:
        if cat in word:
            return cat
    ```
  - Problem: a response like `not human, animal` can be misread as `human`
  - [x] Prefer exact match first
  - [x] Then try first-token match
  - [x] Only then fall back to `other`

- [x] Make `sort_avatars.py` handle missing `out/` gracefully
  - [x] Check `OUT_DIR.exists()` before iterating
  - [x] Exit with a clear message if missing
  - Suggested behavior:
    ```py
    if not OUT_DIR.exists():
        raise SystemExit(f"No output directory found: {OUT_DIR}")
    ```

- [x] Add CLI options to `sort_avatars.py`
  - [x] `--out`, defaulting to `out`
  - Optional but useful, because hardcoded `OUT_DIR = Path("out")` is limiting

## Low priority / cleanup

- [x] Add strict mode to `run.sh`
  - Suggested header:
    ```bash
    #!/bin/bash
    set -euo pipefail
    ```

- [x] Improve dependency install behavior in `run.sh`
  - Current behavior only installs requirements when `venv/` is first created
  - Options:
    - [ ] Always run `pip install -r requirements.txt`
    - [x] Add a `--reinstall` flag
    - [x] Track a requirements hash/timestamp if feeling fancy, but don’t summon Kubernetes

- [x] Remove unused imports
  - [x] `avatar_gen.py`
    - [x] `urllib.request`
    - [x] `urllib.error`
  - [x] `sort_avatars.py`
    - [x] `time`

## Verification checklist

After fixes, run:

- [x] `./venv/bin/python -m py_compile avatar_gen.py sort_avatars.py`
- [x] `./venv/bin/python avatar_gen.py --help`
- [x] `./run.sh --help`
- [x] `./venv/bin/python avatar_gen.py --count 0` should fail clearly
- [x] `./venv/bin/python avatar_gen.py --images-per 0` should fail clearly
- [x] `./venv/bin/python sort_avatars.py --help` if CLI args are added
- [x] `./venv/bin/python sort_avatars.py --out does-not-exist` should fail clearly if `--out` is added

## Notes

No big architecture rewrite needed. This is a single-file slop generator plus a helper script; keep the fixes direct and avoid turning the dinghy into an aircraft carrier.

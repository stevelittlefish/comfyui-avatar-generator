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

- [ ] Make `sort_avatars.py` category parsing stricter
  - Current code checks substring membership:
    ```py
    for cat in CATEGORIES:
        if cat in word:
            return cat
    ```
  - Problem: a response like `not human, animal` can be misread as `human`
  - [ ] Prefer exact match first
  - [ ] Then try first-token match
  - [ ] Only then fall back to `other`

- [ ] Make `sort_avatars.py` handle missing `out/` gracefully
  - [ ] Check `OUT_DIR.exists()` before iterating
  - [ ] Exit with a clear message if missing
  - Suggested behavior:
    ```py
    if not OUT_DIR.exists():
        raise SystemExit(f"No output directory found: {OUT_DIR}")
    ```

- [ ] Add CLI options to `sort_avatars.py`
  - [ ] `--out`, defaulting to `out`
  - Optional but useful, because hardcoded `OUT_DIR = Path("out")` is limiting

## Low priority / cleanup

- [ ] Add strict mode to `run.sh`
  - Suggested header:
    ```bash
    #!/bin/bash
    set -euo pipefail
    ```

- [ ] Improve dependency install behavior in `run.sh`
  - Current behavior only installs requirements when `venv/` is first created
  - Options:
    - [ ] Always run `pip install -r requirements.txt`
    - [ ] Add a `--reinstall` flag
    - [ ] Track a requirements hash/timestamp if feeling fancy, but don’t summon Kubernetes

- [ ] Remove unused imports
  - [ ] `avatar_gen.py`
    - [ ] `urllib.request`
    - [ ] `urllib.error`
  - [ ] `sort_avatars.py`
    - [ ] `time`

## Verification checklist

After fixes, run:

- [ ] `./venv/bin/python -m py_compile avatar_gen.py sort_avatars.py`
- [ ] `./venv/bin/python avatar_gen.py --help`
- [ ] `./run.sh --help`
- [ ] `./venv/bin/python avatar_gen.py --count 0` should fail clearly
- [ ] `./venv/bin/python avatar_gen.py --images-per 0` should fail clearly
- [ ] `./venv/bin/python sort_avatars.py --help` if CLI args are added
- [ ] `./venv/bin/python sort_avatars.py --out does-not-exist` should fail clearly if `--out` is added

## Notes

No big architecture rewrite needed. This is a single-file slop generator plus a helper script; keep the fixes direct and avoid turning the dinghy into an aircraft carrier.

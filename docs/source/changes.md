# What’s new

## 25.12.1-alpha.dev

**Released MONTH YEAR**

### New

- Add Textual GUI for interactive configuration (`--textual` flag) (#79).
- Compute group-level gradient similarity using PCA-based alignment.
- Save DMN similarity summary statistics per group.
- Add `--light-mode` flag to skip age/sex prediction and gradient similarity.
- Expand API reference documentation to cover all public modules.

### Fixes

- Fix light mode handling in `workflow.py` (#ec65351).
- Skip subjects missing from the phenotype file instead of crashing (#77).
- Handle NaN values in t-test for network similarity (#85).
- Fix hardcoded alpha threshold in `significant_level()` — now uses the `alpha` parameter.

### Enhancements

- Simplify age/sex prediction code and make it pre-commit compliant (#89).
- Use loggers instead of print statements throughout the codebase.
- Rename legacy logger (`gc_log` / `giga_connectome`) to `logger` / `wonkyconn`.
- Convert all docstrings to Google format; add missing docstrings to public functions.
- Remove personal debug comments and stale code references.
- Extract magic numbers to module-level constants in `workflow.py`.
- Narrow broad `except Exception` to `except (ValueError, LinAlgError)` for age/sex prediction.
- Fix Textual app mypy errors (Select `NoSelection` handling, `NodeHighlighted` API, focus events).
- Clean up stale commented-out code and misleading comments.
- Use `from scipy import stats` instead of bare `import scipy` in `correlation.py`.

### Changes

- CI: detect file changes and trigger different GitHub Actions workflows (#90).
- Bump `prefix-dev/setup-pixi` from 0.9.3 to 0.9.4 (#93).
- Bump `docker/login-action` from 3.6.0 to 3.7.0 (#94).
- Bump `docker/setup-buildx-action` from 3.11.1 to 3.12.0 (#84).
- Bump `actions/upload-artifact` from 5 to 6 (#83).
- Bump `actions/cache` from 4 to 5 (#82).
- Use `group_by` to infer plot labels instead of hard coding.
- Limit gradient correlation to 3 components.


## 25.12.0-alpha

**Released December 2026**

### New

This release marks the work started by @HippocampusGirl and @htwangtw in 2024,
and later contributions 2025 from
@wangseann (testing, CI, UI),
@claraElk (gradient similarity metrics, debugging),
and @pbergeret12 (prediction on sex and age).

### Fixes

### Enhancements

### Changes

# Utils Refactor Roadmap (Post-P0)

**Status:** P0 and P1 correctness are complete. Branch `feat/refactor-utils`; `uv run pytest -q` → 96 passed, 2 expected warnings.

## How to use this roadmap

Start each new chat with:

> Read `docs/superpowers/plans/2026-10-30-refactor-roadmap.md` and `docs/superpowers/plans/<plan>.md`. Execute the plan with the executing-plans skill. Do not start a different plan in the same session.

Use one plan per session. Stop after the plan's final verification unless the human partner explicitly asks to continue.

## Recommended sequence

1. **P1 correctness** — `2026-10-30-p1-correctness.md` — ✅ complete
2. **ML architecture** — `2026-10-30-ml-architecture.md`
3. **Packaging hygiene** — `2026-10-30-packaging.md` (optional)
4. **Testing and tooling** — `2026-10-30-testing-tooling.md`

Rationale:

- P1 fixes behavior while the current flat modules are still easy to locate.
- Architecture extraction should happen while the P1 tests are fresh; it must not change public behavior.
- Packaging hygiene is deliberately small: dependency floors, requirements generation, and a wheel check. It does **not** rename `utils` or add lazy imports, because those changes add daily-use friction without solving a current problem.
- Testing/tooling finishes with CI, coverage, linting, and a final package build that validates everything above.

Testing/tooling can be moved earlier if CI is wanted before the architecture changes. Packaging hygiene is optional and can be skipped; it no longer changes the import package.

## Plan ownership

| Plan | Owns | Must not do |
|---|---|---|
| P1 correctness | Feature selection, A/B, time series, preprocessing/EDA edge cases, optional-dependency boundaries | Package rename, ML core extraction, plotting redesign |
| ML architecture | Shared ML core, single-pass evaluation, shared tuning/curves, computation/render separation | Package rename, unrelated feature work |
| Packaging hygiene | Dependency floors, optional-extra versions, requirements generation, wheel build verification | Package rename, lazy imports, behavior changes, ML refactors |
| Testing/tooling | pytest-cov, ruff, CI, coverage, optional-extra markers, final package checks | New library features, package rename |

## Global constraints for every plan

- Keep Python **3.14** unless the human partner explicitly changes it.
- Keep `uv` as the dependency manager.
- Use pytest and the `Agg` matplotlib backend; tests must not open windows.
- Do not require optional text/translation extras in default tests.
- Every production change starts with a failing test.
- Preserve public function names unless the plan explicitly includes a migration step.
- Run the full suite before each commit and at the end of the session.
- Update the plan's ledger under `.superpowers/sdd/<plan-basename>/progress.md` as tasks complete.

## Current baseline

- Python 3.14.3
- pandas 2.3.3, scikit-learn 1.7.2, seaborn 0.13.2
- 96 tests passing; two expected warnings (unknown-binary warning and seaborn `vert` deprecation)
- P0 plan: `docs/superpowers/plans/2026-10-29-p0-correctness.md`
- P1 plan: `docs/superpowers/plans/2026-10-30-p1-correctness.md`

## Deferred from the P0 review

Keep these in mind; the relevant plan owns them:

- P1: addressed by `2026-10-30-p1-correctness.md` (numeric/all-NaT date summaries, empty-frame outlier/scaling behavior, encoder completeness, optional-dependency boundaries, plus feature-selection/A-B/time-series fixes).
- Architecture: duplicated classification/regression logic, repeated CV fitting, formatted-string metrics, plot side effects, logging.
- Packaging hygiene: dependency floors, requirements duplication, wheel contents, and description cleanup. The generic `utils` name and eager imports are intentionally retained for notebook ergonomics.
- Testing/tooling: coverage for untested modules, CI, linting, optional-extra test strategy.

# Contributing

This is the research code behind a single paper, so the bar for a change is that every number in the paper stays
reproducible. Issues and pull requests are welcome: bug reports, reproduction reports on other hardware, and
extensions to other models or languages.

## Setup

```bash
make setup            # Python 3.12 environment with the exact pinned versions (needs uv)
make data GATED=1     # released data; the gated parts need approved access on Hugging Face
uvx pre-commit install  # optional: ruff and file checks on every commit
```

## Before opening a pull request

```bash
make lint test        # ruff and the unit tests (CPU)
make verify           # if you touched analysis code: every paper number must be reproduced byte for byte
```

## Conventions

- **Results.** Never type a result into the paper, README or a card. Add a macro in `src/refusal_parity/reporting/paper_numbers/` and
  name its source file.
- **Analysis plan.** Changes to the analysis go into the dated deviations log
  (`research/00_research_plan.md`), labelled confirmatory, exploratory or post hoc.
- **Comparisons** pair runs generated on the same hardware.
- **Commits** follow conventional messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

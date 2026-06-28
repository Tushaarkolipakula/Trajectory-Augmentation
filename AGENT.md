# Instructions

Use Conda environments. Do not change anything system-wide, only install or edit files inside our environment so we can purge and restart easily.
Please use `conda activate qwen-vla` for all project work.
**Rule:** Always run large, long-running scripts (like training or benchmarking) inside a `tmux` session so they are persistent and easy to monitor!

## Strict Execution Rules

1. **Never Kill Processes:** Do NOT ever kill a running process, terminal, or `tmux` session unless explicitly commanded or granted permission by the user. Always ask first.
2. **Never Delete Outputs:** Do NOT ever delete the `outputs/` directory or any files/folders inside of it. Evaluation and training data retention is strictly enforced.

## File Tracking Rules

- **Always track:** code files (.py), configs (.yaml, .toml), docs (.md, .tex), Makefile, .gitignore
- **Never track:** raw data (.hdf5), processed data (.parquet, .csv, .json), checkpoints (.pt, .pth), logs, outputs/, wandb/, caches (__pycache__, .ruff_cache)

## Datasets and Trajectories

- Trajectories and dataset demo HDF5 files are located in `data/LIBERO-datasets` (e.g., `data/LIBERO-datasets/libero_10`).

## Collaboration Hygiene (Required)

Before any push:

- Update `CHANGELOG.md` with all meaningful changes (one bullet per file/module touched).
- Overwrite `CONTEXT.md` with a precise snapshot of the current state.
- Update `INSTRUCTIONS.md` if new decisions or workflows are introduced.

All collaborators must follow this so downstream work is predictable and resumable.

## Keep This File Compact

Target length: ~200 lines. If it grows beyond that, consolidate or move detail to
other docs (e.g., `CHANGELOG.md` or `results/`). Reduce length before pushing.

## Living Documents — Maintain These Every Session

### `changelog.md`

One entry per meaningful change. Format:

```
## YYYY-MM-DD — <short title>
- Added: <what and why>
- Changed: <what and why>
- Deleted: <what and why>
```

Rules: one bullet per file/module touched. No prose. If a change spans multiple
files for one logical reason, group them under one entry. Append only — never edit
past entries. Keep entries concise (aim for 3–6 bullets each).

### `context.md`

A single snapshot of *where the project is right now*. **Overwrite it entirely** after
every few tasks or at any natural stopping point. It is the file a new agent or
collaborator reads first to get up to speed. It must answer:

- What phase are we in and what was just completed?
- What is the next task (T-code) and what does it need?
- What decisions were made that aren't obvious from the code?
- What is currently broken or incomplete?
- What are the current best val metrics (model, dataset, Recall@10)?

Target length: 100–200 lines. Enough detail to resume without reading the full codebase.
No speculation — only facts about the current state.

**Update `changelog.md` for every change. Update `context.md` at the end of every
work session or after completing a phase.**

## Code Style

- **Readable over clever.** If a line needs a comment to be understood, rewrite it.
- **Type-annotated everywhere.** Full type hints on every function. `from __future__ import annotations` at the top of every file.
- **No magic numbers** outside config files.
- **Imports:** stdlib → third-party → local, separated by blank lines. Absolute only.
- **Line length:** 100. **Formatter:** `ruff format`. **Linter:** `ruff check`. Both pass clean before any commit.

### PyTorch

- Shape comment on first use if non-obvious: `# (B, T, D)`.
- Never `.cuda()` directly — resolve device from config or `utils/device.py`.
- `nn.Module`: `__init__` declares submodules only; `forward` is pure computation.
- One class per concept.
- Use `einops` for non-trivial reshaping over chains of `.view()` / `.permute()`.

### Comments

Write only when: (1) a non-obvious algorithmic choice needs one line of justification,
(2) a shape annotation aids clarity, (3) a training phase boundary is entered.
Never restate what the code obviously does.

### Error Handling

- No `try/except` for control flow.
- `assert` for invariants (tensor shapes, config consistency).
- `ValueError` / `RuntimeError` with a descriptive message for user-facing misconfiguration.

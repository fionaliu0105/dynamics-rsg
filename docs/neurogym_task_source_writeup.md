# Env fix + neurogym task-source migration — writeup

Record of the 2026-07-21 work that (a) fixed the broken `rsg` env, (b) established that
neurogym can run locally, and (c) rewired the pipeline to use neurogym as the default
training-data generator. Companion to `docs/env_spike.md` (Solution D), `PLAN_TRACK1.md`
(Blocker A), and the task modules `src/task/rsg.py` / `src/task/rsg_neurogym.py` /
`src/task/__init__.py`.

---

## 1. What the problem originally was

`docs/env_spike.md` documented a dependency conflict, and the `rsg` conda env was actually
broken by it. The root cause is a chain:

- This machine runs **x86-64 (Rosetta) Python on arm64 hardware**. On x86-64 macOS, the
  newest installable PyTorch wheel is **torch 2.2.2**, compiled against **numpy 1.x**.
- **neurogym 2.3.1** declares `numpy==2.2.*`, so installing it dragged the env up to
  **numpy 2.2.6**.
- torch 2.2.2's numpy bridge can't run under numpy 2 → `import torch` threw
  `_ARRAY_API not found` and `torch.from_numpy(...)` raised
  `RuntimeError: Numpy is not available`. **torch was unusable.**

There was also a stray junk file named `2.2.2` in the repo root. The spike doc's original
conclusion was pessimistic: *you can't have one env — use the pure-numpy generator locally,
and neurogym only on the cluster.*

## 2. What was done (in order)

1. **Diagnosed and fixed the broken env** — removed neurogym, pinned numpy to 1.26.4,
   deleted the junk `2.2.2` file (pip output accidentally redirected into a file by an
   unquoted `pip install torch>2.2.2`). torch worked again; 42 tests passed.
2. **Compared `rsg.py` to the real neurogym `ReadySetGo` source** — confirmed it faithfully
   mirrors neurogym's timing, and aligned the one silently-diverging constant
   (`pulse_width` 20 → **83 ms**, neurogym's ready/set period duration).
3. **Built the neurogym-subclass task source** (`src/task/rsg_neurogym.py`) and **re-ran the
   env spike** in an isolated throwaway env — specifically testing whether neurogym really
   needs numpy 2.
4. **Folded the finding into the real `rsg` env** and added a skip-guarded parity test;
   deleted the throwaway env.
5. **Rewired the pipeline** so neurogym is the default data generator, via a config-driven
   dispatch point.

## 3. What was found (the key insights)

- **The stray file was harmless junk** — a shell-redirect accident, not a directory or
  anything the code depended on.
- **The task code was already correct** — `rsg.py` is a faithful, deliberate *extension* of
  neurogym's `ReadySetGo` (two priors, ramp target, Bayesian jitter), not a buggy copy. Only
  one constant needed aligning.
- **The big one:** neurogym's `numpy==2.2.*` pin is **conservative metadata, not a real code
  requirement.** Installed with `--no-deps` under numpy 1.26, it imported and ran perfectly.
  Every *actual* missing piece was a plain pure-python package (`natsort`, `pydantic`,
  `tomlkit`, `tqdm`, `pydantic-settings`, `loguru`) — none of which force numpy 2. **This
  dissolved the "can't have one env" conclusion.** neurogym and x86-mac torch *can* coexist
  locally.
- **The two task generators are byte-identical** — the neurogym subclass and the pure-numpy
  generator produce exactly the same batches (inputs, targets, masks, conditions) at both
  `dt=1` and `dt=5`. Switching between them changes zero data; it only changes which code
  path is the "source of truth."

## 4. Current state

**Environments** — three remain: `base`, `CtDEnv`, and a **unified `rsg`**:

- numpy **1.26.4** (pinned), torch **2.2.2** (working), neurogym **2.3.1** (via `--no-deps`),
  plus the similarity stack (rsatoolbox, POT, DSA, scipy).
- Caveat: `pip check` shows a cosmetic `neurogym requires numpy==2.2.*` **metadata warning**
  — not a runtime problem. neurogym-under-numpy-1 is validated for the API surface the
  subclass actually uses (documented in `env_spike.md` Solution D).

**Code** — task generation is now config-driven with neurogym as the default:

- `src/task/__init__.py` is a **facade**: `from src.task import make_batch, build_trial`
  dispatches on `cfg.task_source` (default `"neurogym"`), falling to the pure-numpy
  `"standalone"` only by explicit opt-in. No silent fallback — if neurogym is selected but
  missing, it raises a clear "install it or use standalone" error.
- `Config.task_source` field; the trainer logs the active backend; `train.py` has a
  `--task-source` flag and prints it.
- Both backends exist: `rsg_neurogym.py` (default) and `rsg.py` (fallback).

**Tests** — **50 passing** in `rsg`. The neurogym-specific tests skip cleanly on an env
without neurogym, so CI never breaks either way.

**Docs** — `env_spike.md` updated with "Solution D" (the numpy<2 + neurogym recipe) and a
fix to a previously-broken command.

**Two things to keep in mind:**

- **Nothing is committed** (as of this writeup) — every change is in the working tree.
- **The RNN training loop itself is still a stub** (`train_one_seed` raises
  `NotImplementedError`). Only the *data-generation* path up to that loop was wired, so when
  the loop is implemented it uses neurogym by default with no further plumbing.

Net effect: the env is fixed, the false assumption that broke it is gone, the two task
sources are provably equivalent, and the pipeline now uses neurogym as its data generator
going forward.

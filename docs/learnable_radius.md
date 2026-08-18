# Parameterized (Learnable) Frequency-Masking Radius

This fork extends SPAI with a **learnable frequency-masking radius**, as a first step
toward physics-informed frequency masking for the medical imaging domain.

## Background

SPAI splits every image patch into a low- and a high-frequency component using a **binary
circular mask** in the 2D-DFT domain, defined by a fixed radius `r`. The original work
hard-codes `r = 16` (`MODEL.FRE.MASKING_RADIUS`), a value the authors found optimal for
natural images because it aligns with the MFM pretext task the backbone was trained on.

In the medical domain this assumption breaks down: medical images are spectrally
homogeneous (≥99.4% of energy sits in the low frequencies) and the high-frequency band is
dominated by acquisition-physics noise (CT Poisson noise, MRI k-space aliasing), not by
synthesis artifacts. A single fixed `r = 16` (covering only ~1.6% of frequency bins) is
unlikely to be the right partition. This feature lets the model **learn** the radius from
data instead.

## What changed

- **`spai/models/filters.py`** — new `SoftCircularMask` module. The hard `distance < r`
  threshold is relaxed into a differentiable sigmoid boundary
  `M(u, v) = sigmoid((r − d(u, v)) / τ)`, so gradients flow into a learnable scalar radius
  `r`. At a low temperature `τ` it reproduces the original binary mask almost exactly.
- **`spai/models/sid.py`** — `MFViT` / `PatchBasedMFViT` accept `learnable_radius` and
  `mask_temperature`. **Critically**, when the learnable radius is enabled the frozen
  backbone is frozen through `requires_grad = False` instead of a `torch.no_grad()` block,
  so the autograd graph of the low/high-frequency streams is preserved and gradients can
  reach the radius. (Under the original `no_grad` freezing the radius would silently
  receive zero gradient.) The original-image stream, which does not depend on the mask, is
  still computed under `no_grad` to save memory.
- **`spai/config.py`** — new keys:
  - `MODEL.FRE.LEARNABLE_MASKING_RADIUS` (default `False` — original behavior unchanged),
  - `MODEL.FRE.MASK_TEMPERATURE` (default `1.0`),
  - `TRAIN.RADIUS_LR` (default `0.01`),
  - `MODEL.FINETUNE_FROM` (path to a full checkpoint to fine-tune from).
- **`spai/optimizer.py`** — the radius parameter gets its **own optimizer group** at
  `TRAIN.RADIUS_LR` with no weight decay. Because the radius is measured in frequency bins
  (a much larger scale than the network weights), it needs a dedicated, higher learning
  rate to move meaningfully.
- **`spai/utils.py` + `spai/__main__.py`** — new `--finetune-from` option initializes the
  **whole** model from a checkpoint (e.g. the released `spai.pth`) with `strict=False`, so
  newly introduced components keep their initialization and obsolete ones are ignored. The
  learned radius is logged to the console, TensorBoard, and Neptune each epoch.
- **Neptune is now optional.** Set `DISABLE_NEPTUNE=1` (or simply don't install/configure
  Neptune) and training falls back to a no-op logger; TensorBoard logging is unaffected.

All changes are backward compatible: with `LEARNABLE_MASKING_RADIUS: False` (the default,
and the setting in `configs/spai.yaml`) the behavior is bit-identical to upstream SPAI.

## Usage

Fine-tune from the released checkpoint with the learnable radius:

```bash
python -m spai train \
  --cfg configs/spai_learnable_radius.yaml \
  --batch-size 4 \
  --data-path datasets/medical.csv \
  --csv-root-dir . \
  --finetune-from weights/spai.pth \
  --output output/learnable_radius \
  --tag exp \
  --amp-opt-level O0 \
  --opt TRAIN.RADIUS_LR 0.01
```

The paired baseline (fixed radius) uses the unmodified `configs/spai.yaml` with the same
`--finetune-from`. Running both on the identical split is the ablation for the thesis.

## A note on GPU memory

With the fixed radius, the frozen backbone runs under `torch.no_grad()`, so no activations
are stored. The learnable radius **must** retain the autograd graph of the low- and
high-frequency streams through all 12 ViT blocks (that is how the gradient reaches `r`),
so training uses meaningfully more memory than the stock frozen path. On a T4 16 GB this is
comfortable at `--batch-size 4` (the smoke-test setting), but if you raise the batch size
and hit OOM, lower `--batch-size` and compensate with `--accumulation-steps`, and/or reduce
`DATA.AUGMENTED_VIEWS` and `MODEL.FEATURE_EXTRACTION_BATCH`. Note that `--use-checkpoint`
(gradient checkpointing) is **not** wired into this backbone, so it is not an available
mitigation. The original-image stream is kept under `no_grad` precisely to limit this cost.

## Fine-tune vs. retrain

**Fine-tuning from `spai.pth` is the correct approach** — the backbone is frozen in all
configurations, so there is nothing to re-learn in it. The released checkpoint already
contains the best-trained SRS projectors, SCV, SCA, and classifier head; the learnable
radius sits *in front of* the frozen backbone and initializes to 16, so loading `spai.pth`
does not invalidate any pretrained weight. Retraining the 180k-image natural-image stage
would only re-derive weights the checkpoint already holds.

## Kaggle

A ready-to-run smoke test is provided at
[`kaggle/learnable_radius_smoke_test.ipynb`](../kaggle/learnable_radius_smoke_test.ipynb).
It clones this repo, downloads `spai.pth` via `gdown`, builds a CSV from
`medical_spai_dataset/`, fine-tunes both the fixed and learnable variants, and reports the
learned radius trajectory. Use the GPU accelerator (T4 x2; only GPU 0 is used) and install
`requirements-kaggle.txt` (APEX is not required — train with `--amp-opt-level O0`).

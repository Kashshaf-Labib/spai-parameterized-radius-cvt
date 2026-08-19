# Physics-Informed Frequency Masking — Learnable Masking Radius

> **Scope of this document.** This is the complete reference for how the SPAI
> frequency-masking radius `r` was turned from a hard-coded constant (`r = 16`) into a
> **learnable parameter** that is fine-tuned end-to-end. This learnable radius is the
> concrete, working first stage of the "physics-informed frequency masking" idea from the
> thesis: it lets the model *learn* where to split the spectrum into low- and
> high-frequency bands instead of assuming the natural-image value of 16. The fuller
> vision — a **per-modality** mask conditioned on CT/MRI/X-ray acquisition physics — is a
> direct extension of this mechanism and is described in the last section.

---

## 1. Why the radius matters (the motivation in one page)

SPAI detects AI-generated images by measuring how *self-consistent* an image is across the
frequency spectrum. To do that it first splits every image into a **low-frequency** part
and a **high-frequency** part using a circular mask in the 2D Fourier domain. The size of
that circle is the **masking radius `r`**. Everything inside radius `r` is "low frequency";
everything outside is "high frequency".

The original authors fixed `r = 16` because it is optimal for **natural photographs**,
whose spectral energy falls off as `1/f^α` with `α ≈ 2.5`. Their ablation (Table 4 of the
paper) confirms 16 is the sweet spot *for that domain*.

Medical images are different:

- Their spectral slope is steeper (`α ≈ 2.87` for chest X-rays) — they are *already* smooth,
  so ≥99.4 % of the energy sits in the low-frequency band.
- The little high-frequency content that exists is mostly **acquisition-physics noise**
  (CT Poisson noise, MRI k-space aliasing, X-ray detector noise), *not* the synthesis
  artifacts SPAI is looking for.

A fixed circle at `r = 16` (only ~1.6 % of the frequency bins) is therefore very unlikely
to be the right place to split a medical spectrum. The idea implemented here is simple and
direct: **make `r` a parameter and let the data choose it.** Because `r` is now learned
jointly with the detection loss, gradient descent moves the low/high boundary to wherever
it produces the most discriminative low-vs-high comparison for the target domain.

---

## 2. The original SPAI architecture (what we are building on)

SPAI has one frozen backbone plus four trainable pieces. The masking radius feeds the very
first step; everything downstream consumes its output. Below is the full pipeline with the
formulas, so the role of `r` is unambiguous.

### 2.1 Frozen spectral backbone `G`

`G` is a **ViT-B/16** pre-trained with **Masked Frequency Modeling (MFM)** — a
self-supervised task where the network learns to reconstruct a masked frequency band of
*real* images. This teaches `G` the spectral distribution of real content. In SPAI, `G`'s
weights are **frozen**; only the pieces below are trained.

- Depth `N = 12` transformer blocks, embedding dim `d = 768`, patch size `16`.
- An image patch of `224×224` becomes `L = (224/16)² = 196` tokens.

**CvT fork adaptation.** The equations below describe the original paper's ViT path. The
CvT configuration in this fork uses all ten blocks of CvT-13's homogeneous final stage:
`N = 10`, `d = 384`, and `L = 196`. Consequently SRS has `6N = 60` values and the
projected spectral vector has `1024 + 60 = 1084` values. The masking, SRS, SCV, SCA, and
classification equations are otherwise unchanged.

### 2.2 Frequency masking — where `r` lives

Given a patch `x ∈ ℝ^{H×W}`, take the 2D DFT and center it:

```
χ = fftshift(FFT2(x))          # complex spectrum, ℂ^{H×W}
```

The paper defines a binary mask by radius `r` from the spectrum center `(c_H, c_W)`
(Eq. 1 of the paper):

```
M_paper(u, v) = 0   if  d((u,v),(c_H,c_W)) < r     (inside the circle)
              = 1   otherwise                       (outside the circle)
```

where `d(·,·)` is Euclidean distance. The low- and high-frequency images are then:

```
x_high = IFFT2( χ ⊙ M_paper )          # keeps OUTSIDE the circle  → high-pass
x_low  = IFFT2( χ ⊙ (1 − M_paper) )    # keeps INSIDE the circle   → low-pass
```

**Code convention (important for reading the source).** The repo's
`filters.generate_circular_mask` builds the *complementary* mask — `1` **inside** the
circle, `0` outside:

```python
mask = torch.where(radius < r, 1, 0)   # 1 inside radius, 0 outside  (= 1 − M_paper)
```

and `filters.filter_image_frequencies(x, mask)` returns `(low, high)` as:

```python
low  = IFFT2( χ ⊙ mask )         # inside kept   → low-pass
high = IFFT2( χ ⊙ (1 − mask) )   # outside kept  → high-pass
```

So the code's `mask` is a **low-pass** mask (1 inside). The end result (`x_low`, `x_high`)
is identical to the paper; only the sign convention of the stored mask differs. Our
learnable mask follows the **code convention** (1 inside), so it is a drop-in replacement.

### 2.3 Spectral Reconstruction Similarity (SRS)

Each of the three images — original `x`, low `x_l`, high `x_h` — is passed through the
frozen `G`. Block `n` produces token features `z_n, z_n^l, z_n^h ∈ ℝ^{L×d}`, which are
projected (per-block learnable projectors `P_n : ℝ^d → ℝ^D`, `D = 1024`) to
`z̄_n, z̄_n^l, z̄_n^h ∈ ℝ^{L×D}`.

SRS is the **token-wise cosine similarity** between two feature maps:

```
λ(z^A, z^B) = (z^A · z^B) / (‖z^A‖ ‖z^B‖)  ∈ [−1, 1]^{L}
```

Three SRS vectors are computed per block:

```
ω_n^{ol} = λ(z̄_n,   z̄_n^l)     # original vs low
ω_n^{oh} = λ(z̄_n,   z̄_n^h)     # original vs high
ω_n^{lh} = λ(z̄_n^l, z̄_n^h)     # low vs high
```

Take the **mean and std** of each over the `L` tokens → 6 scalars per block. Concatenate
over the `N = 12` blocks:

```
z^λ ∈ [−1, 1]^{6N}      (6 × 12 = 72 values)
```

**Intuition:** because `G` models *real* spectra, for a real image the low/high/original
features stay mutually consistent (high `λ`); for an AI-generated image (out-of-distribution
for `G`) they diverge (low `λ`). `r` decides *what counts as low vs high*, so it directly
shapes these similarities.

### 2.4 Spectral Context Vector (SCV)

Not every SRS value is informative for every image (a flat image has no meaningful
high-frequency signal). The SCV captures *what kind of spectral content* the patch has, so
the network can weight the SRS values accordingly.

From the projected `z̄_n`, take mean and std over the `L` tokens → `z' ∈ ℝ^{N×2D}`. With a
learnable spectral map `C ∈ ℝ^{N×D}` and projections `P_1 : ℝ^{2D}→ℝ^D`, `P_2 : ℝ^D→ℝ^D`:

```
C' = P_2( softmax(C) ⊙ P_1(z') )  ∈ ℝ^{N×D}
z^C = Σ_{n=1..N} C'_n              ∈ ℝ^D
```

The per-patch **spectral vector** concatenates context and similarities:

```
z^S = [ z^C ; z^λ ]  ∈ ℝ^{D + 6N}      (1024 + 72 = 1096)
```

### 2.5 Spectral Context Attention (SCA) — any-resolution fusion

A large image is split into `K` patches of `224×224`. Each patch yields its own `z^S_k`;
stack them into `z̄^S ∈ ℝ^{K×(D+6N)}`. A single learnable query `q ∈ ℝ^{D_h}` (`D_h = 1536`)
fuses them in `O(K)` time:

```
A   = softmax( q · (z̄^S W_K)ᵀ / √D_h )  ∈ (0,1)^{1×K}
z^S = ( A · (z̄^S W_V) ) W_O             ∈ ℝ^{D + 6N}
```

with `W_K, W_V ∈ ℝ^{(D+6N)×D_h}`, `W_O ∈ ℝ^{D_h×(D+6N)}`. During **training** `K_train = 4`
augmented views act as the patches; during **inference** the real patches of the
native-resolution image are used.

### 2.6 Classification head and loss

A 3-layer MLP (ReLU, ReLU, sigmoid) maps the image-level `z^S` to `ŷ ∈ (0,1)`; training
minimizes binary cross-entropy:

```
L_cls = BCE(ŷ, y)
```

### 2.7 Where `r` lives in the code (original)

| Item | Location |
| --- | --- |
| Config value | `MODEL.FRE.MASKING_RADIUS = 16` in `config.py` / `configs/spai.yaml` |
| Mask construction | `MFViT.__init__` builds a **frozen `nn.Parameter`** from `filters.generate_circular_mask(img_size, masking_radius)` (`requires_grad=False`) |
| Mask usage | `MFViT.forward` → `filters.filter_image_frequencies(x, self.frequencies_mask)` |
| Backbone freezing | the feature extraction runs inside a `torch.no_grad()` block |

The key facts that make `r` learnable *possible* but *non-trivial*: the mask is a fixed
buffer, and the backbone is frozen by wrapping its forward in `no_grad`. Both must change.

---

## 3. The learnable-radius implementation

### 3.1 `SoftCircularMask` — a differentiable circle

A hard threshold `d < r` has a zero/undefined derivative w.r.t. `r`, so it cannot be
learned by gradient descent. We replace the step with a **sigmoid boundary**
(`spai/models/filters.py`):

```
M_soft(u, v) = σ( (r − d(u,v)) / τ )          # σ = logistic sigmoid
```

- `r` — the **learnable** scalar radius (`nn.Parameter`, initialized to
  `MODEL.FRE.MASKING_RADIUS = 16`).
- `d(u,v)` — precomputed Euclidean distance of each frequency bin from the center
  (a non-trainable buffer, not saved in checkpoints).
- `τ` — the **temperature** (`MODEL.FRE.MASK_TEMPERATURE`, default `1.0`), the width of the
  soft transition band in frequency bins.

Properties:

- **Inside** the circle (`d < r`): `r − d > 0` → `σ(·) → 1` (low-pass keeps it).
- **Outside** (`d > r`): `r − d < 0` → `σ(·) → 0`.
- As `τ → 0`, `M_soft` converges to the exact binary `generate_circular_mask` (verified in
  the unit tests to >99.9 % agreement). So the soft mask is a strict, differentiable
  generalization of the original.
- Same convention as the code's `generate_circular_mask` (1 inside), so it slots directly
  into `filter_image_frequencies` with no downstream change.

```python
class SoftCircularMask(nn.Module):
    def __init__(self, input_size, initial_radius, temperature=1.0):
        self.temperature = float(temperature)
        self.radius = nn.Parameter(torch.tensor([float(initial_radius)]))   # learnable r
        distances = linalg.vector_norm(generate_centered_2d_coordinates_grid(input_size), dim=-1)
        self.register_buffer("distances", distances, persistent=False)      # fixed d(u,v)

    def forward(self):
        return torch.sigmoid((self.radius - self.distances) / self.temperature)
```

### 3.2 The critical fix: letting the gradient reach `r`

This is the single most important part of the implementation.

For `∂L/∂r` to exist, the autograd graph must stay intact along:

```
L  →  SRS  →  z̄^l, z̄^h  →  G(x_l), G(x_h)  →  x_l, x_h  →  IFFT ⊙ M_soft  →  r
```

But the original code freezes the backbone by running `G` under `torch.no_grad()`, which
**severs this graph** — `r` would receive *zero* gradient and never move, while the run
would *appear* to work. Simply removing `no_grad` is also wrong: the backbone weights would
then start updating (SPAI requires them frozen).

The fix (`spai/models/sid.py`, `MFViT.forward`): when the learnable radius is enabled,
freeze the backbone by setting **`requires_grad = False` on the ViT parameters** (so its
weights never update) while **keeping the graph** for the two frequency-filtered streams:

```python
if self.frozen_backbone:
    if self.learnable_radius and torch.is_grad_enabled():
        with torch.no_grad():
            x = self.vit(x)          # original stream does NOT depend on r → no graph (saves memory)
        low_freq = self.vit(low_freq)  # low  stream keeps graph → gradient can reach r
        hi_freq  = self.vit(hi_freq)   # high stream keeps graph → gradient can reach r
    else:
        with torch.no_grad():          # fixed-radius path: unchanged from upstream
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
```

Summary of who is differentiable:

| Tensor / weights | Fixed radius (original) | Learnable radius (this work) |
| --- | --- | --- |
| Backbone ViT weights | frozen via `no_grad` (params still `requires_grad=True`) | frozen via `requires_grad=False` |
| Original-image stream `G(x)` | `no_grad` | `no_grad` (independent of `r`) |
| Low/high streams `G(x_l), G(x_h)` | `no_grad` | **graph kept** (so `∂L/∂r` exists) |
| Masking radius `r` | not a parameter | **trainable** |

This was validated on CPU: `r` receives a finite non-zero gradient, an optimizer step moves
it (16.0000 → 16.0100), and the backbone parameters receive **no** gradient.

### 3.3 Optimizer: `r` gets its own learning rate

`r` is measured in **frequency bins** — a much larger numerical scale than the network
weights. At the backbone/head learning rate (`5e-4`) it would barely move. So
`spai/optimizer.py` places `r` in its **own parameter group** with a dedicated learning
rate and no weight decay:

```python
if name.endswith("soft_frequency_mask.radius"):
    group = {"lr": config.TRAIN.RADIUS_LR, "weight_decay": 0.0, ...}   # RADIUS_LR default 0.01
```

All other trainable parameters (SRS projectors, SCV map, SCA query/weights, classifier)
keep the original layer-decayed AdamW scheme at `BASE_LR`. Because the backbone is now
`requires_grad=False`, its parameters are automatically excluded from the optimizer in the
learnable path (a nice side effect: the optimizer only holds what actually trains).

The timm cosine scheduler scales **each group by its own base LR**, so `r`'s group warms up
to `RADIUS_LR` and cosine-decays independently of the head groups (verified). During the
`WARMUP_EPOCHS` the radius LR ramps from ~0 up to `RADIUS_LR`, then decays over the run.

### 3.4 Configuration knobs

Added in `config.py` (all backward-compatible; defaults reproduce upstream behavior):

| Key | Default | Meaning |
| --- | --- | --- |
| `MODEL.FRE.LEARNABLE_MASKING_RADIUS` | `False` | Turn the learnable radius on/off. `False` = original fixed binary mask. |
| `MODEL.FRE.MASK_TEMPERATURE` | `1.0` | Sigmoid transition width `τ` (bins). Smaller ⇒ sharper, closer to binary. |
| `MODEL.FRE.MASKING_RADIUS` | `16` | Initial value of `r` (unchanged meaning; now the starting point). |
| `TRAIN.RADIUS_LR` | `0.01` | Learning rate for the `r` parameter group. |
| `MODEL.FINETUNE_FROM` | `''` | Full checkpoint (e.g. `spai.pth`) to initialize the whole model for fine-tuning. |

The CvT configs are `configs/spai_cvt.yaml` and
`configs/spai_cvt_learnable_radius.yaml`. The original ViT counterparts remain available
as `configs/spai.yaml` and `configs/spai_learnable_radius.yaml`.

### 3.5 Fine-tuning from `spai.pth` (checkpoint compatibility)

This subsection applies only to the legacy **ViT** configuration. The released
`spai.pth` cannot initialize CvT-SPAI because its backbone and all dimension-dependent
phase-two layers have incompatible shapes.

`--finetune-from weights/spai.pth` loads the **entire** released model with
`strict=False` (`spai/utils.py: load_finetune_checkpoint`). Because the learnable model:

- **has** a new key `mfvit.soft_frequency_mask.radius` (absent from `spai.pth`) → it keeps
  its initialization of `16`, and
- **lacks** the old key `mfvit.frequencies_mask` (present in `spai.pth`) → that entry is
  simply ignored,

loading is clean: only the radius is "missing" (kept at 16) and only the obsolete fixed
mask is "unexpected". Every trained weight (projectors, SCV, SCA, classifier) transfers
intact. This is why **fine-tuning, not retraining, is correct** — the backbone is frozen in
all configurations, so there is nothing in it to relearn; `r` sits *in front of* the frozen
backbone and starts from the natural-image optimum of 16.

### 3.6 Logging

The current radius is written to the console, TensorBoard (`train/masking_radius`), and
Neptune once per epoch, and appears in the per-iteration training log line
(`masking_radius 16.xxxx`). It is also recoverable directly from any checkpoint:

```python
torch.load(ckpt, map_location="cpu")["model"]["mfvit.soft_frequency_mask.radius"].item()
```

---

## 4. How fine-tuning through the variable `r` actually works

Step by step, for one CvT training run with
`configs/spai_cvt_learnable_radius.yaml` and
`--pretrained weights/cvt_mfm_pretrain.pth`:

1. **Build model.** `MODEL.FRE.LEARNABLE_MASKING_RADIUS=True` ⇒ `MFViT`/`PatchBasedMFViT`
   create a `SoftCircularMask(img_size=224, initial_radius=16, temperature=1.0)` and set the
   CvT backbone to `requires_grad=False` and keeps its BatchNorm layers in eval mode.
2. **Load weights.** The 455 phase-one CvT encoder entries are loaded strictly; the
   phase-two projectors, SCV, SCA, classifier, and `r` retain their initialization.
3. **Build optimizer.** `r` → its own group at `RADIUS_LR=0.01`; SRS/SCV/SCA/classifier →
   layer-decayed AdamW at `BASE_LR=5e-4`; backbone → excluded (frozen).
4. **Each training step** (`train_one_epoch`), for each of the 4 augmented views:
   - Recompute the soft mask `M_soft` from the *current* `r`.
   - FFT the view, multiply by `M_soft` / `(1−M_soft)`, IFFT → `x_l`, `x_h`.
   - Run the frozen `G` on `x`, `x_l`, `x_h` (low/high keep the graph).
   - Compute SRS → SCV → spectral vector → SCA → classifier → `ŷ`.
   - `L = BCE(ŷ, y)`; `L.backward()` sends gradients to the trainable heads **and to `r`**
   (through the IFFT and the frozen CvT of the low/high streams).
   - `optimizer.step()` nudges `r` by its own learning rate; the heads by theirs.
5. **Per epoch:** validate, checkpoint (best val loss or `--save-all`), and log the new `r`.
6. **Over the run:** `r` migrates from 16 to whatever value maximizes the discriminative
   power of the low-vs-high comparison for the target (medical) domain.

The paired **baseline** run is identical but uses `configs/spai_cvt.yaml`
(`LEARNABLE_MASKING_RADIUS=False`), so `r` stays pinned at 16. Comparing the two on the same
data isolates the effect of freeing `r` — this is the ablation for the thesis.

---

## 5. Side-by-side comparison

| Aspect | Original SPAI | This implementation |
| --- | --- | --- |
| Masking radius `r` | hard-coded constant `16` | trainable scalar `nn.Parameter`, init `16` |
| Mask shape | binary step: `1` inside, `0` outside | soft sigmoid: `σ((r−d)/τ)`, → binary as `τ→0` |
| Mask storage | frozen buffer `frequencies_mask` | recomputed each forward from `r` |
| Backbone freezing | `torch.no_grad()` in forward | `requires_grad=False` (keeps graph for `r`) |
| Gradient to `r` | n/a | flows through IFFT + frozen ViT (low/high streams) |
| Optimizer | one AdamW scheme | + dedicated `r` group at `RADIUS_LR`, no decay |
| New config keys | — | `LEARNABLE_MASKING_RADIUS`, `MASK_TEMPERATURE`, `RADIUS_LR`, `FINETUNE_FROM` |
| Init for training | `--pretrained` (phase-one backbone) | `--pretrained` for CvT phase one; `--finetune-from` only for a matching full phase-two model |
| Default behavior | — | **identical to upstream** when the flag is `False` |
| Downstream (SRS/SCV/SCA/head) | unchanged | unchanged |

---

## 6. Symbol and hyperparameter reference

| Symbol | Meaning | Value |
| --- | --- | --- |
| `r` | masking radius (learnable) | init 16 |
| `τ` | sigmoid temperature (`MASK_TEMPERATURE`) | 1.0 |
| `d(u,v)` | distance of bin `(u,v)` from spectrum center | fixed |
| `N` | selected backbone blocks | ViT: 12; CvT: 10 |
| `d` | backbone embedding dim | ViT: 768; CvT: 384 |
| `L` | tokens per 224×224 patch | 196 |
| `D` | projection dim (`PROJECTION_DIM`) | 1024 |
| `D_h` | SCA hidden dim (`ATTN_EMBED_DIM`) | 1536 |
| `K_train` | augmented views used as patches in training | 4 |
| `z^λ` | SRS feature vector | ViT: `ℝ^{72}`; CvT: `ℝ^{60}` |
| `z^S` | spectral vector `[z^C; z^λ]` | ViT: `ℝ^{1096}`; CvT: `ℝ^{1084}` |
| `BASE_LR` | LR for heads | 5e-4 |
| `RADIUS_LR` | LR for `r` | 0.01 |
| `EPOCHS` / `WARMUP_EPOCHS` | schedule | 35 / 5 (full) |

Core formulas:

```
M_soft(u,v) = σ( (r − d(u,v)) / τ )                 # learnable circular mask
x_low  = IFFT2( fftshift(FFT2(x)) ⊙ M_soft )        # low-pass
x_high = IFFT2( fftshift(FFT2(x)) ⊙ (1 − M_soft) )  # high-pass
λ(a,b) = (a·b)/(‖a‖‖b‖)                              # token-wise SRS (cosine)
z^S    = [ z^C ; z^λ ]                              # per-patch spectral vector
L_cls  = BCE(ŷ, y)                                  # training objective
```

---

## 7. File-by-file change map

| File | Change |
| --- | --- |
| `spai/models/filters.py` | New `SoftCircularMask` (learnable `r`, sigmoid boundary). |
| `spai/models/sid.py` | `MFViT`/`PatchBasedMFViT` accept `learnable_radius`, `mask_temperature`; `requires_grad=False` freezing; graph-preserving low/high forward; `get_frequencies_mask`, `get_masking_radius`; wired through `build_mf_vit`. |
| `spai/config.py` | `LEARNABLE_MASKING_RADIUS`, `MASK_TEMPERATURE`, `RADIUS_LR`, `FINETUNE_FROM`. |
| `spai/optimizer.py` | Dedicated `r` param group at `RADIUS_LR`. |
| `spai/utils.py` | `load_finetune_checkpoint` (full model, `strict=False`). |
| `spai/__main__.py` | `--finetune-from`; per-epoch/-iteration radius logging; optional Neptune. |
| `configs/spai_cvt_learnable_radius.yaml` | Ready-to-run CvT learnable-radius config. |
| `tests/models/test_filters.py`, `tests/models/test_sid.py` | Unit tests for the mask and gradient flow. |
| `docs/learnable_radius.md` | Short usage/how-to (companion to this file). |

---

## 8. How to run

Learnable radius (ours):

```bash
python -m spai train \
  --cfg configs/spai_cvt_learnable_radius.yaml \
  --batch-size 4 \
  --data-path datasets/medical.csv \
  --csv-root-dir . \
  --pretrained weights/cvt_mfm_pretrain.pth \
  --output output/cvt_learnable_radius \
  --tag exp \
  --amp-opt-level O0 \
  --opt TRAIN.RADIUS_LR 0.01
```

Fixed-radius baseline (same command, `configs/spai_cvt.yaml`). Run both on the same split; the
difference is the ablation. Watch the `masking_radius` in the logs drift off 16.

---

## 9. What comes next: per-modality physics-informed masking

The single global `r` here is the mechanism; the full physics-informed mask from the thesis
extends it in a natural way:

- **Modality conditioning.** Replace the single scalar `r` with a small MLP that maps a
  modality one-hot (CT / MRI / X-ray / Pathology) to a per-modality radius (or a per-frequency
  attention profile). This needs a `modality` column in the dataset CSV.
- **Physics supervision.** Ground the learned mask in the empirical spectral profiles of
  known-real images of each modality, so it *down-weights* the acquisition-noise bands
  (CT Poisson, MRI k-space aliasing, X-ray detector noise) instead of mistaking them for
  synthesis evidence.

Both reuse the exact gradient-flow and optimizer machinery documented above — the learnable
`SoftCircularMask` simply becomes a learnable, modality-indexed mask generator. Everything
downstream (SRS, SCV, SCA, classifier) stays unchanged, just as it does here.

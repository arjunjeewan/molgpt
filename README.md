# MolGPT
In this work, we train small custom GPT on Moses and Guacamol dataset with next token prediction task. The model is then used for unconditional and conditional molecular generation. We compare our model with previous approaches on the Moses and Guacamol datasets. Saliency maps are obtained for interpretability using Ecco library.

- The processed Guacamol and MOSES datasets in csv format can be downloaded from this link:

>~~https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing~~

>https://www.kaggle.com/datasets/virajbagal/ligflow-datasets (working link)

- Original Guacamol dataset can be found here:

>https://github.com/BenevolentAI/guacamol

- Original Moses dataset can be found here:

>https://github.com/molecularsets/moses

- All trained weights can be found here:

>https://www.kaggle.com/virajbagal/ligflow-final-weights (old weights, prior to modernization)


To train the model, make sure you have the datasets' csv file in the same directory as the code files.

# Training

```
./train_moses.sh
```

```
./train_guacamol.sh
```

# Generation

```
./generate_guacamol_prop.sh
```

```
./generate_moses_prop_scaf.sh
```

If you find this work useful, please cite:

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): MolGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1 

---

# Architecture Modernization (RoPE + SwiGLU + RMSNorm)

The transformer-decoder was modernized with three contemporary components and the
effect of each was attributed via controlled, multi-seed ablations on MOSES and
GuacaMol. Components are **param-matched** (modern ≈6.35M vs baseline ≈6.38M) and
reach the **same training loss** (baseline 0.310–0.319, modern 0.309–0.316), so every
difference below is an *inductive-bias* effect, not a capacity or fit difference.

- **RoPE** — replaced the learned absolute positional embedding (rotation on Q/K of
  SMILES tokens only; prepended condition tokens left unrotated).
- **SwiGLU** — bias-free `down(SiLU(gate(x))·up(x))` MLP, hidden=682, param-matched.
- **RMSNorm** — replaced all three LayerNorms.

**Headline:** modernization is *RoPE wearing a SwiGLU/RMSNorm coat*. RoPE alone carries
both the **novelty gain** and the **FCD distribution-match penalty** (a clean trade-off);
SwiGLU and RMSNorm are behaviorally neutral, cost-free modernizations. Pick by use case —
**de-novo / novelty → modern; benchmark / FCD-matching → baseline.** Full writeup, repro
steps, and caveats in [`MODERNIZATION.md`](MODERNIZATION.md).

## Setup (environment)

Create the conda env from [`environment.yml`](environment.yml), then run the two
post-install steps that can't be expressed declaratively (both are documented inline in
the file — `molsets` needs `--no-deps`, and `guacamol` needs a one-line scipy patch):

```bash
conda env create -f environment.yml
conda activate molgpt

# (1) MOSES metrics — must be --no-deps (molsets would otherwise downgrade numpy/pandas/scipy)
pip install molsets==0.3.1 --no-deps

# (2) patch guacamol for modern scipy (scipy.histogram was removed; it equals numpy.histogram)
sed -i 's/from scipy import histogram/from numpy import histogram/' \
  "$(python -c 'import guacamol,os;print(os.path.dirname(guacamol.__file__))')/utils/chemistry.py"
```

The `torch` pin is the CUDA 12.6 wheel — edit the `--extra-index-url`/`+cu126` tag in
`environment.yml` for a different CUDA (or CPU). **Datasets are not bundled**: rebuild
`datasets/moses2.csv` with `python experiments/preprocess_moses.py`, and fetch
`datasets/guacamol2.csv` from Kaggle (`virajbagal/ligflow-datasets`).

## Training & evaluating the modernized configurations

All commands run from the inner `molgpt/` code directory (the one containing `train.py`
and `experiments/`), with the env active (`conda activate molgpt`). Checkpoints are
written to `../cond_gpt/weights/`. (The `experiments/run_*.sh` drivers `cd` to this
directory automatically, so they can be launched from anywhere once the env is active.)

**MOSES — architecture ablation** (`--arch {baseline,modern}`):

```bash
python experiments/train_seeded.py --arch modern   --seed 1 --epochs 10 --run_name unconditional_moses_modern_s1
python experiments/train_seeded.py --arch baseline --seed 1 --epochs 10 --run_name unconditional_moses_baseline_s1
```

**GuacaMol — per-component ablation** (`--config {baseline,rope,swiglu,rmsnorm,modern}`):

```bash
# unconditional (single component on, or 'modern' for all three)
python experiments/train_ablate.py --config rope --seed 42 --epochs 10 --run_name guaca_uncond_rope_s42

# property-conditioned
python experiments/train_ablate.py --config modern --seed 42 --epochs 10 --run_name guaca_logp_modern_s42 \
    --num_props 1 --props logp

# scaffold-conditioned
python experiments/train_ablate.py --config modern --seed 42 --epochs 10 --run_name guaca_scaf_modern_s42 \
    --scaffold

# scaffold + property
python experiments/train_ablate.py --config modern --seed 42 --epochs 10 --run_name guaca_scaflogp_modern_s42 \
    --num_props 1 --props logp --scaffold
```

**Evaluation** (add `--baseline` for baseline-arch MOSES checkpoints; modern is the default):

```bash
# MOSES decoding-temperature frontier (fast metrics) -> datasets/sweep_<tag>.csv
python experiments/sweep_decode.py --ckpt ../cond_gpt/weights/unconditional_moses_modern_s1.pt \
    --temps "0.7 0.9 1.0 1.2 1.6" --out datasets/sweep_modern_s1.csv

# MOSES full metrics (FCD/SNN/Frag/Scaf/...) at one temperature -> datasets/moses_metrics_<tag>_T1.0.json
python experiments/gen_eval_moses.py --ckpt ../cond_gpt/weights/unconditional_moses_modern_s1.pt \
    --temp 1.0 --tag modern_s1

# GuacaMol (config + conditioning are auto-read from the checkpoint's sidecar .json)
python experiments/eval_guaca.py --run_name guaca_uncond_rope_s42 --temp 1.0
```

**End-to-end, idempotent pipelines** (each phase skips already-finished work, so re-running is safe):

```bash
# MOSES multi-seed: train seeds 1,2 (both archs) + 25-epoch modern -> eval all -> aggregate
bash experiments/run_train_all.sh
bash experiments/run_eval_all.sh
python experiments/aggregate_seeds.py        # -> experiments/multiseed_results.md

# GuacaMol full matrix: train -> eval -> aggregate
bash experiments/run_guaca_all.sh            # -> experiments/guaca_ablation_results.md
```

To widen the GuacaMol matrix, edit `A_SEEDS` / `B_SEEDS` / `B_MODES` in
`experiments/run_train_guaca.sh` and re-run `run_guaca_all.sh`.

### Table 1 — GuacaMol per-component ablation (unconditional, T=1.0, 3 seeds {1,2,42})

Each component on (✓) / off (✗). Validity/Unique/Novelty/KL-div: higher = better.
FCD: lower = better (distance to held-out reference). **The trade-off isolates to RoPE
alone** (FCD 0.992→1.253, ≈10σ, zero seed overlap); SwiGLU/RMSNorm are FCD-neutral.

| Config | RoPE | SwiGLU | RMSNorm | Validity | Unique | Novelty | KL-div | FCD |
|---|:--:|:--:|:--:|--:|--:|--:|--:|--:|
| baseline | ✗ | ✗ | ✗ | 0.977±0.001 | 0.999±0.000 | 0.955±0.001 | 0.994±0.001 | 0.992±0.007 |
| +RoPE | ✓ | ✗ | ✗ | 0.981±0.001 | 0.999±0.000 | 0.960±0.003 | 0.988±0.001 | **1.253±0.026** |
| +SwiGLU | ✗ | ✓ | ✗ | 0.977±0.002 | 0.999±0.000 | 0.947±0.002 | 0.995±0.000 | 1.004±0.009 |
| +RMSNorm | ✗ | ✗ | ✓ | 0.979±0.001 | 0.999±0.000 | 0.953±0.001 | 0.994±0.000 | 0.969±0.024 |
| modern (all) | ✓ | ✓ | ✓ | 0.981±0.001 | 0.999±0.000 | 0.949±0.002 | 0.989±0.001 | 1.336±0.025 |

### Table 2 — Full MOSES metrics at T=1.0, baseline vs modern (3 seeds {1,2,42})

The FCD trade-off reproduces on a second benchmark: modern is ~10σ worse on FCD/Test
(0.571→0.906) while fast quality metrics stay identical.

| Metric | baseline | modern |
|---|--:|--:|
| valid | 0.9945±0.0004 | 0.9937±0.0004 |
| unique@10000 | 0.9990±0.0004 | 0.9983±0.0004 |
| Novelty | 0.7845±0.0058 | 0.7896±0.0050 |
| IntDiv | 0.8504±0.0004 | 0.8503±0.0002 |
| **FCD/Test** | **0.5709±0.0351** | **0.9059±0.0289** |
| SNN/Test | 0.6241±0.0005 | 0.6231±0.0025 |
| Frag/Test | 0.9972±0.0001 | 0.9849±0.0007 |
| Scaf/Test | 0.8833±0.0029 | 0.8808±0.0042 |
| Filters | 0.9978±0.0002 | 0.9974±0.0002 |
| FCD/TestSF | 1.1634±0.0364 | 1.3137±0.0240 |

### Table 3 — MOSES decoding frontier (3 seeds {1,2,42})

The validity "gap" is a **decoding-temperature artifact, not architectural** — both archs
are ≥0.978 valid at T≤1.2 and collapse to ~0.85 at T=1.6. Report at T=1.0–1.2.

| Arch | T | Validity | Unique | Novelty | IntDiv1 |
|---|--:|--:|--:|--:|--:|
| baseline | 0.7 | 0.9994±0.0002 | 0.9933±0.0006 | 0.6640±0.0039 | 0.8327±0.0007 |
| baseline | 0.9 | 0.9977±0.0002 | 0.9981±0.0007 | 0.7426±0.0046 | 0.8447±0.0011 |
| baseline | 1.0 | 0.9945±0.0003 | 0.9990±0.0002 | 0.7797±0.0027 | 0.8496±0.0010 |
| baseline | 1.2 | 0.9783±0.0009 | 0.9995±0.0003 | 0.8454±0.0002 | 0.8586±0.0008 |
| baseline | 1.6 | 0.8601±0.0060 | 0.9996±0.0001 | 0.9292±0.0013 | 0.8707±0.0005 |
| modern | 0.7 | 0.9996±0.0003 | 0.9870±0.0008 | 0.6906±0.0047 | 0.8317±0.0006 |
| modern | 0.9 | 0.9973±0.0006 | 0.9965±0.0010 | 0.7625±0.0064 | 0.8447±0.0007 |
| modern | 1.0 | 0.9944±0.0008 | 0.9982±0.0006 | 0.7932±0.0021 | 0.8501±0.0006 |
| modern | 1.2 | 0.9794±0.0022 | 0.9992±0.0000 | 0.8502±0.0059 | 0.8588±0.0006 |
| modern | 1.6 | 0.8481±0.0040 | 0.9995±0.0001 | 0.9292±0.0028 | 0.8705±0.0002 |

### Table 4 — Paired novelty edge (modern − baseline), matched by seed

Modern is significantly more novel at low temperature (all 3 per-seed deltas positive);
the edge washes out by T≥1.2. With n=3 the t-test is low-power, so sign-consistency is
the primary evidence.

| T | per-seed deltas | mean Δ | t | p (paired) |
|--:|---|--:|--:|--:|
| 0.7 | +0.0201, +0.0268, +0.0330 | +0.0266 | 7.14 | 0.019 |
| 0.9 | +0.0168, +0.0207, +0.0223 | +0.0199 | 12.25 | 0.007 |
| 1.0 | +0.0087, +0.0151, +0.0167 | +0.0135 | 5.51 | 0.031 |
| 1.2 | +0.0102, −0.0013, +0.0054 | +0.0048 | 1.43 | 0.290 |
| 1.6 | +0.0025, −0.0033, +0.0008 | +0.0000 | 0.00 | 0.999 |

### Table 5 — Conditional / scaffold generation (GuacaMol, T=1.0, seed 42)

Where the unconditional story is a trade-off, conditional generation is
**neutral-to-better** for modernization. MAD = mean abs. deviation of generated property
from target (lower = better); scaffold-match = fraction matching the conditioning scaffold.

| Mode | Arch | Validity | Unique | Novelty | logP MAD | Scaffold-match |
|---|---|--:|--:|--:|--:|--:|
| prop:logp | baseline | 0.969 | 1.000 | 0.978 | 0.224 | — |
| prop:logp | modern | 0.972 | 0.999 | 0.971 | **0.200** | — |
| scaffold | baseline | 0.994 | 0.794 | 0.992 | — | 0.973 |
| scaffold | modern | 0.995 | 0.801 | 0.989 | — | **0.982** |
| scaffold+logp | baseline | 0.989 | 0.827 | 0.999 | 0.192 | 0.979 |
| scaffold+logp | modern | 0.989 | 0.814 | 0.999 | 0.199 | 0.962 |

### Table 6 — Longer run refutes the undertraining hypothesis (25 vs 10 epochs, modern, seed 42)

More epochs → validity ↑ but novelty ↓ with FCD unchanged, i.e. it moves *away* from the
paper's joint numbers. The residual gap vs the manuscript is data/preprocessing/reporting,
not training budget; the FCD gap is genuinely architectural.

| T | metric | modern 10ep (seed-mean) | modern 25ep |
|--:|---|--:|--:|
| 1.0 | valid | 0.9944 | 0.9963 |
| 1.0 | novelty | 0.7932 | 0.7346 |
| 1.2 | valid | 0.9794 | 0.9856 |
| 1.2 | novelty | 0.8502 | 0.7988 |
| 1.6 | valid | 0.8481 | 0.8890 |
| 1.6 | novelty | 0.9292 | 0.8894 |



# JEPA Location World Model — Project Specification

## Overview

This project applies **JEPA (Joint Embedding Predictive Architecture)** to human mobility data. It treats a user's sequence of location check-ins as a temporal signal — like video frames but with geographic cells over time — and learns to predict future location representations from past context, entirely self-supervised (no labeled destinations).

The core question: **Given where someone has been, can we predict where they will go next?**

---

## Architecture

### How JEPA Works (Simple Explanation)

Traditional models predict raw outputs (e.g., "the next cell is g_3889_10296"). JEPA instead predicts **representations** — it learns what the *embedding* of the next location should look like, not the location itself.

```
Trajectory:  [cell_1, cell_2, cell_3 | cell_4]
              <-- context (75%) -->   <- target (25%) ->

              +------------------+
  context --> | Context Encoder  |--> ctx_repr --+
              | (4-layer Tfm)    |               |
              +------------------+               v
                                          +-----------+
                                          | Predictor |--> predicted repr
                                          | (2-layer) |
                                          +-----------+
                                                |
                                          compare (loss)
                                                |
              +------------------+              v
  all tokens >| Target Encoder   |--> tgt_repr (ground truth)
              | (EMA copy,frozen)|
              +------------------+
```

1. **Tokenizer**: Each check-in becomes a 128-dim vector = `cell_embedding + hour_embedding + dow_embedding + position_encoding`
2. **Context Encoder**: 4-layer Transformer that sees only the first 75% of the trajectory
3. **Target Encoder**: Identical architecture but frozen — updated as a slow-moving average (EMA) of the context encoder. Sees all tokens.
4. **Predictor**: Smaller 2-layer Transformer that takes context representations and predicts what the target representations should be at masked positions
5. **Loss**: Smooth L1 (Huber) between predicted and target representations in latent space

The EMA (Exponential Moving Average) update prevents collapse — without it, both encoders could learn to output the same constant vector for everything.

### Classification Head (Fine-tuning Extension)

After JEPA pretraining, a linear classification head (`d_model -> vocab_size`) is added on top of the predictor output and trained with cross-entropy loss while keeping all encoders frozen. This converts latent predictions into actual cell probabilities.

```
predictor output at target positions --> Linear(128, 53207) --> softmax --> cell probabilities
```

---

## Data Pipeline

### Source

**Gowalla** check-in dataset from Stanford SNAP (https://snap.stanford.edu/data/loc-gowalla.html).

| Stat | Value |
|------|-------|
| Total check-ins | ~6.4M |
| Unique users | ~107K |
| Date range | Feb 2009 - Oct 2010 |

### Processing Steps

```
1. Load raw check-ins (user_id, timestamp, lat, lng)
2. Filter inactive users (< 10 check-ins)
3. Quantize (lat, lng) -> geocell ID using 0.01-degree grid (~1km cells)
     e.g., (38.897, -77.036) -> "g_3889_10296"
4. Group into 7-day non-overlapping trajectory windows per user
5. Filter short trajectories (< 2 tokens)
6. Split by USER (80/10/10) -- no user appears in multiple splits
7. Apply k-anonymity: cells visited by < 5 users -> <UNK>
8. Build vocabulary from train split only
9. Export as JSONL
```

### Output Data

| File | Records | Description |
|------|---------|-------------|
| `train.jsonl` | 525,229 | Training trajectories |
| `val.jsonl` | 64,981 | Validation trajectories |
| `test.jsonl` | 65,110 | Test trajectories |
| `vocab.json` | 53,207 entries | Geocell to integer index mapping |

### Record Format

```json
{
  "user_id": "u_a1b2c3d4e5f6",
  "date": "2010-04-03",
  "tokens": [
    {"geocell": "g_3889_10296", "hour": 8,  "dow": 2},
    {"geocell": "g_3890_10297", "hour": 12, "dow": 2},
    {"geocell": "g_3889_10296", "hour": 18, "dow": 2}
  ]
}
```

- `user_id`: Salted SHA-256 hash (never raw account IDs)
- `geocell`: Integer grid ID (never raw lat/lng)
- `hour`: 0-23
- `dow`: 0-6 (Monday = 0)

---

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 128 | Embedding dimension |
| `enc_layers` | 4 | Context/Target encoder Transformer layers |
| `pred_layers` | 2 | Predictor Transformer layers |
| `n_heads` | 4 | Attention heads |
| `max_len` | 64 | Max sequence length |
| `ctx_frac` | 0.75 | 75% context, 25% target |
| `CELL_SIZE_DEG` | 0.01 | ~1km geocell resolution |
| `vocab_size` | 53,207 | Distinct geocells + PAD + UNK |
| `EMA momentum` | 0.996 -> 0.999 | Ramps linearly over training |
| `optimizer` | AdamW | lr=1e-3, weight_decay=0.01 |
| `lr schedule` | Warmup + cosine | 500-step warmup |
| `batch_size` | 64 | Training batch size |
| `epochs` | 30 | JEPA pretraining epochs |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 accuracy** | Fraction where the #1 predicted cell matches the true next cell |
| **Top-5 accuracy** | Fraction where the true cell is among the top 5 predictions |
| **Median km error** | Median haversine distance between predicted and true cell centroids |
| **P90 km error** | 90th percentile distance error (tail performance) |
| **emb_std** | Mean per-dimension std of target representations (collapse health; healthy ~ O(1)) |

---

## Baselines

Four baselines establish the floor and ceiling for the task:

| Baseline | Logic |
|----------|-------|
| **Popularity (global)** | Always predict the single most-visited cell across all users |
| **Popularity (per-user)** | Predict the most-visited cell for the specific user |
| **Recency (last cell)** | Predict the last cell in the context sequence |
| **Markov (1st order)** | argmax P(next_cell \| current_cell) from training transition counts |

All baselines are fit on the training split only and evaluated with identical metrics on the test split.

---

## Results History

### Run 1: JEPA Pretraining + Latent-NN Evaluation (2026-05-22 to 2026-05-25)

**Training**: 30 epochs on CPU, best checkpoint saved at epoch 23 (val_loss = 0.1478).

**Training health metrics** (at convergence):
- Loss: ~0.24
- emb_std: ~0.93 (healthy, no collapse)
- pred_target_cos: ~0.68 (reasonable but not great)

**Evaluation method**: Nearest-neighbor search in latent space -- encode all 53K cells through the target encoder, then find the cell whose representation is closest to the predictor's output.

| Model | Top-1 | Top-5 | Median km | P90 km |
|-------|-------|-------|-----------|--------|
| Popularity (global) | 0.137 | 0.137 | 0.0 | 0.0 |
| Popularity (per-user) | 0.137 | 0.137 | 0.0 | 0.0 |
| Recency (last cell) | 0.237 | 0.237 | 3.1 | 58.9 |
| **Markov (1st order)** | **0.244** | **0.408** | **2.8** | **52.6** |
| JEPA (latent-NN) | 0.017 | 0.035 | 599.1 | 8320.4 |

**Outcome**: JEPA performed catastrophically worse than all baselines. 14x lower accuracy than Markov, 200x worse distance error.

**Root cause analysis**:

1. **No classification objective**: JEPA optimizes smooth L1 loss in representation space, not cross-entropy over cell IDs. The nearest-neighbor evaluation is an indirect proxy the model was never trained for.
2. **53K-way nearest-neighbor is too hard**: Markov searches ~5-20 observed successors per cell; JEPA searches all 53,207 cells with no filtering.
3. **Hour/dow mismatch**: Cell embeddings for the lookup table used `hour=0, dow=0`, but query trajectories have real temporal values. Since embeddings are summed (`cell + hour + dow`), this creates systematic offset.
4. **Short sequences**: Average trajectory length ~4 tokens. With 75% context, the Transformer sees ~3 tokens -- too little for attention patterns to outperform direct transition counting.

### Run 2: Classification Head Fine-tuning (2026-06-04)

**Approach**: Freeze all JEPA encoders. Add a linear head (`128 -> 53,207`) trained with cross-entropy loss directly on next-cell prediction. 10 epochs, lr=3e-3 with cosine decay.

**Epoch progression**:

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1 | 0.244 | 0.227 | 0.0354 |
| 2 | 0.272 | 0.227 | 0.0351 |
| 3 | 0.283 | 0.227 | 0.0348 |
| 5 | 0.306 | 0.232 | 0.0342 |
| 8 | 0.341 | 0.244 | 0.0333 |
| 10 | 0.357 | 0.244 | 0.0332 |

**Final evaluation (test split)**:

| Model | Top-1 | Top-5 | Median km | P90 km |
|-------|-------|-------|-----------|--------|
| Popularity (global) | 0.137 | 0.137 | 0.0 | 0.0 |
| Popularity (per-user) | 0.137 | 0.137 | 0.0 | 0.0 |
| Recency (last cell) | 0.237 | 0.237 | 3.1 | 58.9 |
| **Markov (1st order)** | 0.244 | 0.408 | **2.8** | **52.6** |
| JEPA (latent-NN) | 0.017 | 0.035 | 599.1 | 8320.4 |
| **JEPA + cls head** | **0.246** | **0.426** | 3.3 | 82.7 |

**Outcome**: JEPA + cls head **beats Markov on both Top-1 (24.6% vs 24.4%) and Top-5 accuracy (42.6% vs 40.8%)**. Median km error is close (3.3 vs 2.8) but P90 tail is worse (82.7 vs 52.6) due to occasional confident but geographically distant predictions.

**Analysis**: A single frozen linear layer on top of JEPA representations already surpasses the strongest counting-based baseline on accuracy, confirming that the JEPA pretraining learned genuinely useful mobility representations. The distance tail is worse because the model has no geographic inductive bias -- it can predict any of the 53K cells regardless of distance.

**Next steps to improve further**:
- End-to-end fine-tuning (unfreeze encoders + predictor)
- Geographic candidate filtering (only consider nearby cells)
- Add time-gap embeddings for temporal awareness

---

## File Structure

```
jepa-location/
|-- SPEC.md                          # This file
|-- shared/
|   +-- geocell-contract.md          # Quantization + sequence schema
+-- py-model/
    |-- data/
    |   |-- raw/                     # Gowalla raw data (not committed)
    |   |-- processed/               # train/val/test JSONL + vocab.json
    |   |-- load_gowalla.py          # Load + clean Gowalla check-ins
    |   |-- geocell.py               # Deterministic lat/lng -> cell ID
    |   |-- build_sequences.py       # Full pipeline: quantize -> split -> export
    |   |-- splits.py                # User-level train/val/test splits
    |   +-- dataset.py               # PyTorch Dataset + DataLoader
    |-- baselines/
    |   |-- popularity.py            # Global + per-user most-frequent cell
    |   |-- recency.py               # Predict last cell
    |   +-- markov.py                # 1st-order transition counts
    |-- model/
    |   |-- tokenizer.py             # Cell + hour + dow + positional embeddings
    |   |-- encoder.py               # Transformer encoder (shared architecture)
    |   |-- predictor.py             # Smaller Transformer predictor with mask tokens
    |   +-- jepa.py                  # Full JEPA: encoders + predictor + EMA + loss
    |-- train.py                     # JEPA pretraining loop
    |-- finetune.py                  # Classification head fine-tuning
    |-- eval.py                      # Baselines + JEPA evaluation
    |-- visualize.py                 # UMAP embedding visualization
    |-- checkpoints/
    |   |-- best.pt                  # Best JEPA pretrained checkpoint (epoch 23)
    |   |-- best_cls.pt              # Best classification head checkpoint
    |   +-- umap_embeddings.png      # UMAP plot of learned embeddings
    |-- requirements.txt
    +-- tests/
        +-- test_model.py
```

---

## How to Run

```bash
cd jepa-location/py-model
source venv/Scripts/activate    # or venv/bin/activate on Linux/Mac

# Step 1: Build data (only needed once)
python -m data.build_sequences

# Step 2: JEPA pretraining (30 epochs, ~hours on CPU)
python -m train

# Step 3: Fine-tune classification head (10 epochs)
python -m finetune

# Step 4: Evaluate all models
python -m eval

# Step 5: Visualize embeddings
python -m visualize
```

---

## Privacy

| Rule | Enforcement |
|------|-------------|
| No raw coordinates in exports | JSONL contains geocell IDs + coarse time only |
| Pseudonymous user IDs | Salted SHA-256 hash, salt not exported |
| k-anonymity on cells | Cells with < 5 distinct users -> `<UNK>` |
| No identity targets | Model predicts cells only, never demographic/identity |

---

## Potential Improvements

| Improvement | Expected Impact | Complexity |
|-------------|----------------|------------|
| End-to-end fine-tuning (unfreeze encoders) | High | Low |
| Longer sequences (overlapping windows, user history concat) | High | Medium |
| Hierarchical geocells (multi-resolution H3) | Medium | Medium |
| Time-gap embeddings (delta between check-ins) | Medium | Low |
| User-conditioned prediction (user embeddings) | Medium | Low |
| Trajectory flow graph prior (GETNext-style) | High | High |
| Candidate filtering (only search geographic neighbors) | Medium | Low |

---

## Related Research

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| T-JEPA: Trajectory Similarity via JEPA | `2406.12913` | Most directly related -- JEPA on trajectories for similarity, not prediction |
| V-JEPA 2: Self-Supervised Video World Models | `2506.09985` | Foundational JEPA architecture this project is based on |
| Pre-trained Transformer for Human Mobility | `2406.04029` | Masked Trajectory Modeling -- closest comparable approach |
| GETNext: Trajectory Flow Map Transformer | `2303.04741` | Widely-cited next-POI baseline using flow graphs |
| UniMob: Universal Mobility Prediction | `2412.15294` | SOTA universal mobility with diffusion transformer |
| LAW: Latent World Model (ICLR 2025) | `2406.08481` | Self-supervised world model predicting future scene latents |
| Drive-JEPA: JEPA for Driving | `2601.22032` | JEPA for autonomous driving trajectories |
| Causality-Aware Next Location | `2503.18179` | Causal reasoning for next-location prediction |

**Key gap**: No prior work applies JEPA specifically to next-location prediction. T-JEPA uses JEPA for trajectory *similarity* (a different task). This project occupies a novel intersection.

# Speech Codec Probing from Semantic and Phonetic Perspectives

Reference code for:

> **Speech Codec Probing from Semantic and Phonetic Perspectives.**
> Xuan Shi, Chang Zeng, Tiantian Feng, Shih-Heng Wang, Jianbo Ma, Shrikanth Narayanan.
> arXiv:2603.10371.

We probe four neural speech codecs — **EnCodec** (12 kbps), **DAC**, **MIMI**, **MIMO** —
for the linguistic information their per-layer features carry:

| Result | Experiment | Method | Notebook |
|---|---|---|---|
| **Fig 2** | Semantic / phonetic word-pair probing | per-layer Euclidean distance (synonym vs near-homophone vs random) | `01` |
| **Fig 3** | Articulatory probing | per-layer PWCCA vs vocal-tract distance (rtMRI) | `02` |
| **Fig 4** | MIMI semantic vs accumulated-acoustic | PWCCA, semantic layer split out | `02` |
| **CKA table** | Speech ↔ text alignment | linear CKA of codec features vs LLM text embeddings | `03` |

## Layout

```
├── scripts/          check_env, prepare_librispeech, compute_vtd_batch,
│                     build_fig2_cache, download_data
├── src/
│   ├── codecs/       one file per codec; unified extract_hidden_states()
│   ├── data/         LibriSpeech word-pair construction + samplers
│   ├── vtd/          vocal-tract distance from rtMRI contours
│   ├── analysis/     Euclidean pair distances, PWCCA, CKA, RSA
│   └── config.py     per-experiment settings (env-var path overrides)
├── notebooks/        01–03 run the experiments; 04 redraws all figures from cache
└── data/             shipped caches (figures reproduce with no raw data)
```

## Setup

With **uv** (fast — recommended on a fresh machine / bolt):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .            # Tier 0 + EnCodec/MIMI feature extraction
uv pip install -e ".[all]"     # also DAC, the data-prep tools, and RSA
```

Or with conda + pip:

```bash
conda create -n codec_probing python=3.11 -y && conda activate codec_probing
pip install -r requirements.txt
python scripts/check_env.py    # verify codec backends load and extract
```

## Reproducing the paper

**Figures from the shipped caches — CPU only, no raw data (< 1 min):**

```bash
jupyter nbconvert --to notebook --execute notebooks/04_paper_figures.ipynb
```

This reads `data/euclidean_cache/*.pkl` (Fig 2), `data/output_stats_cache/cca_*.json`
(Fig 3/4) and `data/cka_results.pkl` (CKA), all shipped in the repo.

**Recompute an experiment (Tier 1 — GPU; codecs download from HuggingFace):**

| Notebook | Output | Extra inputs |
|---|---|---|
| `01_semantic_phonetic_probing.ipynb` | `data/euclidean_cache/{codec}.pkl` | shipped `data/word_pairs/librispeech.df.pkl` |
| `02_articulatory_probing.ipynb` | `data/output_stats_cache/cca_*.json` | `data/output_vtd/` (Zenodo) + `SPAN_SPEECH_DIR` audio |
| `03_speech_text_alignment.ipynb` | `data/cka_results.pkl` | paired LLM checkpoints |

Set machine-specific locations via environment variables (no paths are hard-coded):

```bash
export SPAN_SPEECH_DIR=/path/to/SPAN          # parent of <sub_id>/2drt/audio/<track>_audio.wav
export MIMO_CHECKPOINT=/path/to/MiMo-Audio-Tokenizer
```

**From scratch (Tier 2):**

```bash
# LibriSpeech (dev/test-clean) + MFA TextGrids -> word DataFrame + maps
python scripts/prepare_librispeech.py --librispeech-dir <wavs> --textgrid-dir <MFA> --output-dir data/word_pairs
# rtMRI MATLAB contours + grids -> per-track VTD .npy
python scripts/compute_vtd_batch.py --contour-dir <mat> --grid-dir <grids> --output-dir data/output_vtd
```

## Codec backends

All four codecs expose the same API, `extract_hidden_states(audio_path) -> (feats[L, T, D], T, D)`,
where `feats[k]` is the decoded output **accumulated through residual quantizer layers 0..k**
(paper Sec 2.3). Per-codec layer counts: **EnCodec 16, DAC 32, MIMI 32, MIMO 20.**

| Codec | Backend | Install |
|---|---|---|
| EnCodec | `transformers.EncodecModel` (`facebook/encodec_24khz`) | core deps |
| MIMI | `transformers.MimiModel` (`kyutai/mimi`) | core deps |
| DAC | `descript-audio-codec` | `uv pip install -e ".[dac]"` |
| MIMO | vendored `mimo_audio_tokenizer` | `pip install -e <MiMo-Audio-Tokenizer>` + `flash-attn` |

## Data

* **Shipped in git** (small): `data/word_pairs/` (LibriSpeech word DataFrame + synonym/homophone
  maps, stored as plain types), `data/euclidean_cache/`, `data/output_stats_cache/`,
  `data/cka_results.pkl`. These are enough for `notebooks/04`.
* **`data/output_vtd/`** (1.1 GB derived VTD arrays) — fetched from Zenodo:
  `python scripts/download_data.py` (see `data/MANIFEST.md` for the DOI + checksums).
* **SPAN rtMRI speech corpus** — license-gated, not redistributed. Point `SPAN_SPEECH_DIR`
  at your local copy (layout `<sub_id>/2drt/audio/<track>_audio.wav`).

`data/euclidean_cache/*.pkl` were produced from the per-seed distance outputs with
`scripts/build_fig2_cache.py`; `notebooks/01` regenerates the same cache from scratch.

## Attribution

The word-pair probing methodology follows Choi et al., *"Self-Supervised Speech
Representations are More Phonetic than Semantic"* (Interspeech 2024); `src/analysis/pwcca.py`
adapts google/svcca; VTD operates on USC SPAN rtMRI segmentation output. See `NOTICE`.

## Citation

```bibtex
@article{shi2026codec,
  title  = {Speech Codec Probing from Semantic and Phonetic Perspectives},
  author = {Shi, Xuan and Zeng, Chang and Feng, Tiantian and Wang, Shih-Heng and Ma, Jianbo and Narayanan, Shrikanth},
  journal = {arXiv preprint arXiv:2603.10371},
  year   = {2026}
}
```

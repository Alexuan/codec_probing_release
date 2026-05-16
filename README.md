# Speech Codec Probing — Code Release

Reference implementation for the experiments in:

> **Speech Codec Probing from Semantic and Phonetic Perspectives.**
> Xuan Shi, Chang Zeng, Tiantian Feng, Shih-Heng Wang, Jianbo Ma, Shrikanth Narayanan.
> arXiv:2603.10371 (2026).

The paper probes four neural speech codecs — **EnCodec**, **DAC**, **MIMI**, **MIMO** — for the linguistic information their per-layer features carry, via three complementary experiments.

## Layout

```
codec_articulatory_release/
├── scripts/                 # data preparation + env sanity (run as plain .py)
│   ├── check_env.py
│   ├── prepare_librispeech.py
│   └── compute_vtd_batch.py
├── notebooks/               # the three probing experiments + figure reproduction
│   ├── 01_semantic_phonetic_probing.ipynb   # Exp 1, Fig 2
│   ├── 02_articulatory_probing.ipynb        # Exp 2, Fig 3 & 4
│   ├── 03_speech_text_alignment.ipynb       # Exp 3, CKA table
│   └── 04_paper_figures.ipynb               # one-shot Fig 2/3/4 reproduction from cache
├── src/                     # importable library
│   ├── codecs/              # one file per codec (encodec / dac / mimi / mimo)
│   ├── data/                # LibriSpeech word-pair construction
│   ├── vtd/                 # vocal-tract distance from rtMRI
│   ├── analysis/            # PWCCA, CKA, Euclidean pair distances
│   └── config.py            # per-experiment dataclasses
├── data/                    # input/output (caches + word_pairs/output_vtd shipped)
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/check_env.py            # verify codec backends load
```

If you only want to reproduce the figures from the shipped caches, open `notebooks/04_paper_figures.ipynb` directly — it does not need the source datasets.

### Reproducing each experiment

| Step | Command | Output |
|---|---|---|
| 1. Build LibriSpeech word DataFrame + pair maps | `python scripts/prepare_librispeech.py --librispeech-dir <wavs> --textgrid-dir <MFA> --output-dir data/word_pairs` | `data/word_pairs/librispeech.{df,wordmap}.pkl` |
| 2. (optional) Recompute VTD from raw rtMRI tracks | `python scripts/compute_vtd_batch.py --contour-dir <mat> --grid-dir <grids> --output-dir data/output_vtd` | `data/output_vtd/*_vtd.npy` |
| 3. Exp 1 — semantic / phonetic probing | open `notebooks/01_semantic_phonetic_probing.ipynb` | `data/euclidean_cache/<codec>.pkl` |
| 4. Exp 2 — articulatory PWCCA | open `notebooks/02_articulatory_probing.ipynb` | `data/output_stats_cache/cca_*.json` |
| 5. Exp 3 — speech ↔ text CKA | open `notebooks/03_speech_text_alignment.ipynb` | `data/cka_results.pkl` |
| 6. Final figures | open `notebooks/04_paper_figures.ipynb` | Fig 2 / 3 / 4 |

## Codec backends

| Codec | Backend | Install |
|---|---|---|
| EnCodec | HuggingFace `transformers.EncodecModel` (`facebook/encodec_24khz`) | `pip install transformers` |
| DAC | `descript-audio-codec` | `pip install descript-audio-codec audiotools` |
| MIMI | HuggingFace `transformers.MimiModel` (`kyutai/mimi`) | `pip install transformers` |
| MIMO | Vendored `mimo_audio_tokenizer` | `pip install -e /path/to/MiMo_Audio_Tokenizer` plus `pip install flash-attn --no-build-isolation` |

Each codec is implemented in its own file under `src/codecs/`. All four expose the same `extract_hidden_states(audio_path) → np.ndarray[L, T, D]` API where `feats[k]` is the *accumulated* decoded output through RVQ layers `0..k` — matching the paper's Sec 2.3 feature definition.

## Data preparation

### LibriSpeech word pairs (`scripts/prepare_librispeech.py`)

The core construction is a **verbatim port** of
[juice500ml/phonetic_semantic_probing](https://github.com/juice500ml/phonetic_semantic_probing)
(Apache 2.0), which the paper's semantic / phonetic experiment is built on.
The ported code lives in `src/data/_juice500ml.py` with the original
function names preserved to make diffs against the upstream repo trivial.
The pipeline:

1. Parse MFA TextGrids → one row per word occurrence (`build_librispeech_dataframe`).
2. CMU Pronouncing Dictionary → per-word phoneme sequence (stress markers stripped).
3. WordNet → candidate synonyms.
4. Synonym map: keep WordNet pairs whose phoneme-WER is **above** the threshold (0.4 by default) to avoid confounding with homophones.
5. Homophone map: phoneme-WER strictly **between 0 and threshold**, excluding any pair already in the (unfiltered) synonym set.

### Vocal Tract Distance (`scripts/compute_vtd_batch.py`)

For each rtMRI track from the 75-Speaker corpus, intersects MATLAB-tracked articulator contours with a 120-gridline cross-sectional grid spanning lips → larynx, yielding a `(n_frames, 120)` distance array.

## Vendored / re-used third-party code

* `src/analysis/pwcca.py` — Projection-Weighted CCA, adapted from [google/svcca](https://github.com/google/svcca) (Apache 2.0; original header preserved).
* `src/data/_juice500ml.py` — direct port of the LibriSpeech word-pair construction in [juice500ml/phonetic_semantic_probing](https://github.com/juice500ml/phonetic_semantic_probing) (Apache 2.0; original function names and signatures preserved).

## Citation

```bibtex
@inproceedings{shi2026codec,
  title  = {Speech Codec Probing from Semantic and Phonetic Perspectives},
  author = {Shi, Xuan and Zeng, Chang and Feng, Tiantian and Wang, Shih-Heng and Ma, Jianbo and Narayanan, Shrikanth},
  year   = {2026},
  eprint = {2603.10371},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
}
```

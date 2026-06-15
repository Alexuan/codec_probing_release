# Data manifest

## Shipped in this repository (small caches — enough for `notebooks/04_paper_figures.ipynb`)

| Path | Contents |
|---|---|
| `word_pairs/librispeech.df.pkl` | per-word-occurrence DataFrame (LibriSpeech dev/test-clean, MFA-aligned; plain types) |
| `word_pairs/librispeech.wordmap.pkl` | `{synonym_map, homophone_map}` |
| `euclidean_cache/{encodec,dac,mimi,mimo}.pkl` | Fig 2 — per-layer word-pair Euclidean distances (`{synonym,homophone,random} -> [n_seeds, L]`, per-seed mean over pairs; notebook 04 averages across seeds) |
| `output_stats_cache/cca_similarity_across_layers_{encodec_24k_12bps,dac_24k,mimi,mimo}.json` | Fig 3 — per-layer PWCCA means |
| `output_stats_cache/cca_similarity_across_layers_mimi_separate.json` | Fig 4 — MIMI semantic vs accumulated acoustic |
| `cka_results.pkl` | CKA table (MIMI, MIMO) |

Per-codec layer counts: EnCodec 16, DAC 32, MIMI 32, MIMO 20.

## Hosted externally (not in git)

| Item | Size | Source |
|---|---|---|
| `output_vtd/` (`*_vtd.npy`, 460 tracks) | ~1.1 GB | Zenodo — DOI: `TODO`. Fetch with `python scripts/download_data.py --url <zenodo-file-url>`. sha256: `TODO`. |

## Not redistributed (license-gated)

* **SPAN rtMRI 75-Speaker speech corpus** — obtain from USC SPAN; point `SPAN_SPEECH_DIR`
  at your copy. Expected layout: `<sub_id>/2drt/audio/<trackname>_audio.wav`.

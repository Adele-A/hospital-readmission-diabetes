# Train/test split and cross-validation (Stage 4)

Hold-out: `GroupShuffleSplit` by `patient_nbr`, test_size=0.2, seed=42. The test set is not touched until Stage 6.

| split | encounters | patients | positives | positive % |
| --- | ---: | ---: | ---: | ---: |
| train | 79,682 | 55,989 | 9,139 | 11.47 |
| test | 19,658 | 13,998 | 2,175 | 11.06 |

**Patient overlap between train and test: 0** (must be 0).

## Cross-validation inside train — StratifiedGroupKFold (5 folds, grouped by `patient_nbr`)

| fold | train enc | train pat | val enc | val pat | val positive % | group overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 63,746 | 44,846 | 15,936 | 11,143 | 11.47 | 0 |
| 2 | 63,744 | 44,810 | 15,938 | 11,179 | 11.48 | 0 |
| 3 | 63,746 | 44,799 | 15,936 | 11,190 | 11.46 | 0 |
| 4 | 63,746 | 44,764 | 15,936 | 11,225 | 11.47 | 0 |
| 5 | 63,746 | 44,737 | 15,936 | 11,252 | 11.46 | 0 |

All fold group overlaps are 0 — no patient appears in both a fold's train and validation parts. See `tests/test_leakage.py` for the automated checks.

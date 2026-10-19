# Data contract

No official data are bundled. Access is gated at https://huggingface.co/datasets/aims-foundations/measurement-db.

Required CSV columns:

| Column | Meaning |
| --- | --- |
| benchmark_id | Stable benchmark identifier used exclusively for splitting/reporting |
| item_id | Stable item ID within its benchmark; repeated runs keep the same ID |
| subject_id | Full evaluated system configuration ID; not just a model name |
| item_text | Question text available before the target system responds |
| label | Observed binary result, exactly 0 or 1 |
| provider | Provider metadata; use an explicit documented missing-category string if appropriate |

Do not include answers, explanations produced by the evaluated model, grading outputs or response-derived features in item_text. Preserve raw data and provenance; normalized data must not silently overwrite the source. Record download date, pinned revision, source file hashes, join keys, excluded rows and reasons, release eligibility, and benchmark licensing. Repeated runs are legitimate rows; only remove proven accidental export duplicates, with an audit trail.

The supplied generator creates 8 artificial benchmarks, 80 items per benchmark and 8 systems (5,120 rows). It exists only to exercise code and has no relationship to the official data distribution.

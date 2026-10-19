# ColdCast Checkpoint I

The real-data pilot and updated **five-page IEEE report** are complete. Public posting and verification remain pending. Deadline: October 20, 2026, before class.

- [IEEE LaTeX source](ColdCast_Checkpoint_I_IEEE.tex)
- [Five-page PDF](output/pdf/ColdCast_Checkpoint_I_IEEE.pdf)
- [Measured results and reproduction commands](REAL_RESULTS.md)
- [Individual contribution record](CONTRIBUTIONS.md)

## Completed experiments

The pilot contains 51,751 responses and 1,345 normalized text groups across five measurement-db sources. Eleven predictors were evaluated with five held-out-source folds, a secondary item split, and three neural seeds. The pipeline includes frozen MiniLM embeddings, PyTorch fusion, training-only retrieval, temperature calibration, and random-label adaptation on 122 eligible system–source pairs. Seventeen tests and an independent saved-model inference check passed.

Metadata logistic regression achieved item-held-out Brier 0.1794. Constant 0.5 was strongest on source holdout (0.2500); fusion achieved 0.2738. Adaptation improved metadata predictions from 0.2474 to 0.1629 at 31 labels on a separate eligible evaluation pool. These are offline pilot results, not official competition scores.

## Report and publication

The standalone IEEEtran source incorporates real-data methods, results, limitations, and verified activity versus assigned responsibilities. Edit it in the built-in LaTeX editor. The old Markdown/ReportLab report and builder are superseded and must not be used to regenerate the current report.

The root webpage links to the updated PDF. To finish submission, restore GitHub write access if needed, publish the intended report/code/aggregate results, and verify the public PDF before class. Exclude raw data, model caches, credentials, and unrelated local changes. No live publication is claimed.

## Remaining research

Broaden source coverage, evaluate longer-context representations and system-family holdout, repeat acquisition trajectories, compare active policies, and implement/validate the official competition package. These are later research tasks, not prerequisites for reporting this checkpoint's measured findings.

The original synthetic fixture remains a software test. Current results and commands are in REAL_RESULTS.md; data and model versions are pinned in their manifests. Raw data and trained model caches stay local.

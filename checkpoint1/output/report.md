# ColdCast: Calibrated Cold-Start Prediction of AI System Performance
**Project Checkpoint I | CS5814 | October 19, 2026**

Yi Chen Kuo, Sudip Manchare, Jaideep Manupati, Akshada Shailesh Rane
Department of Computer Science, Virginia Tech

**DRAFT FOR TEAM REVIEW - individual contributions require completion.**

## Abstract
ColdCast studies whether an AI system will answer a new evaluation question correctly before that system is executed. The proposed framework combines question representations, system metadata, retrieval evidence, and limited revealed outcomes to estimate calibrated correctness probabilities. This report extends the original three-page proposal with a concrete validation protocol, an executable preliminary implementation, measured software-check results, a transparent progress assessment, and an updated work plan. The new CPU prototype compares rate-based predictors, text and metadata logistic models, and a two-hidden-layer fusion network. A reproducible synthetic run verifies training, probability output, and calibration; it does not establish performance on the official benchmark. Access to the official dataset, pretrained semantic encoding, retrieval, and adaptation remain outstanding. The checkpoint therefore reports implementation progress and testable hypotheses rather than claiming a completed competition system.

## 1. Motivation and research question
Aggregate benchmark accuracy can hide differences between system capability and question difficulty. An inexpensive predictor of correctness could help allocate evaluation effort, identify uncertain cases for human review, and compare candidate systems before running every benchmark item. Such a predictor must generalize to new content and report useful probabilities, not merely rank familiar examples.

The central question remains: do question semantics, system metadata, retrieval evidence, and limited adaptation improve out-of-benchmark correctness prediction over simple historical rates? Our primary hypothesis is that combining item and system information reduces Brier error relative to either modality alone. A separate hypothesis is that calibration improves probability quality under distribution shift. Neither hypothesis is established by the present synthetic run.

The task is binary prediction, with y equal to one for an observed correct response. For system s and item i, the desired output is p(s,i) = P(y=1 | attributes and permitted evidence). No target response, grading explanation, or hidden evaluation label may enter the features. Operational deployment would additionally require evidence that errors and uncertainty are acceptable for the intended decision context.

## 2. Related work and intended contribution
Truong et al. [2] investigate amortized evaluation using item content to estimate difficulty, motivating the use of semantic information when labels are scarce. Zhuang et al. [3] connect AI evaluation with psychometrics and adaptive testing. Jiang et al. [4] examine reliability limitations of item-response methods in AI benchmark regimes. Shang et al. [5] combine contextual representations with multidimensional latent capabilities, while identifying cross-scenario generalization as a challenge.

ColdCast's intended contribution is a controlled comparison of these complementary signals under one leakage-aware evaluation design. The project does not claim that feature concatenation itself is novel. Its research value depends on fair baselines, credible held-out evaluation, informative ablations, and reproducible calibration analysis. The citation metadata have been checked against the current paper records; the original proposal's author initials and the current title of reference [3] have been corrected.

<!-- PAGEBREAK -->
## 3. Data contract and evaluation design
The intended source is the AIMS measurement-db release [6]. Access currently requires an account-level approval process; no official dataset has been downloaded for this checkpoint. A reproducible data release must be pinned by revision and file hashes. The ingestion record will identify join keys, missing-value handling, exclusions, and permitted use. Archived records must be distinguished from active public releases.

The prototype accepts a normalized CSV containing benchmark_id, item_id, subject_id, item_text, label, and provider. Each row represents an observed response. Repeated runs of the same subject-item pair remain together across partitions. The subject identifier must distinguish complete evaluated configurations. Benchmark and item identifiers are used for grouping and reporting, never as predictive features. Unknown subject/provider values map to reserved categories.

### 3.1 Four distinct partitions
Whole benchmarks are assigned to training, development, calibration, and test sets using a saved seed and manifest. Training learns parameters and category vocabularies; development selects the training epoch; calibration selects a positive scalar temperature; test is reserved for final measurement. This strengthens the proposal's item-held-out design because unseen items from a familiar benchmark can still share task-specific patterns with training examples. Item-held-out results will be a secondary diagnostic, explicitly distinguished from benchmark-held-out results.

Exact duplicate normalized text across benchmarks triggers an error rather than allowing silent leakage. Near-duplicates, templated items, and overlapping source datasets require further audit before real experiments. With few benchmarks, repeated random splits do not create independent evidence; results should state which benchmarks were held out and how sensitive conclusions are to that choice.

### 3.2 Measures and comparison rules
Brier score is the mean squared difference between predicted probability and binary outcome. Log loss penalizes confidently wrong predictions. ECE is a diagnostic computed here with ten equal-width probability bins; each reliability point records the average confidence, observed correctness, and sample count. A low ECE alone does not prove useful predictions. We report both response-weighted and equally weighted subject-benchmark Brier summaries when applicable.

For preliminary uncertainty analysis, the prototype bootstraps whole test items, keeping their subjects and repeated responses together. Intervals describe variation conditional on the selected test benchmark; they do not capture uncertainty about transfer to arbitrary new benchmarks. Real-data analysis will use multiple held-out benchmarks and report the number of independent groups supporting each conclusion. All baselines receive identical partitions and permitted information.

### 3.3 Update to the external competition protocol
The current competition [1] evaluates benchmark cold start at label budgets 0, 1, 3, 7, 15, and 31, rather than the proposal's fixed five-label setup. Acquisition and scoring pools are separate. Its primary measure is pair-macro Brier ALC, with budget weights 0.1, 0.2, 0.2, 0.2, 0.2, and 0.1. The present prototype measures zero-label Brier only. Competition packaging and acquisition logic must be checked against the current starting kit before any submission; the original offline-only runtime description is outdated.

<!-- PAGEBREAK -->
## 4. Preliminary implementation and executed checks
All new scripts and outputs reside in checkpoint1/. The implementation uses NumPy so the initial pipeline runs on CPU without GPU setup or paid APIs. A stable 128-dimensional signed token hash represents text; training-only one-hot subject/provider categories represent metadata. These are inexpensive prototype features, not pretrained semantic embeddings. The proposed learned metadata embeddings and frozen sentence encoder remain future work.

The fusion network concatenates the two inputs and uses hidden layers of 64 and 32 tanh units, followed by a sigmoid output. Full-batch Adam minimizes binary cross-entropy with weight decay of 0.0001, learning rate 0.01, and a maximum of 120 epochs. The epoch with the lowest development Brier score is retained. A separate calibration set selects temperature from 81 log-spaced values between 0.2 and 5 by log loss. No calibration bias is fitted in this prototype.

### 4.1 Synthetic verification only
The artificial dataset contains 5,120 binary responses from eight systems and 640 items across eight invented benchmarks. Its generator combines system ability, task difficulty, and stochastic outcomes. Seed 5814 yields 3,200 training rows and 640 rows each for development, calibration, and test. Every partition has all eight systems, so this run checks new-benchmark handling, not unseen-system generalization. Synthetic dependencies were deliberately made learnable and cannot validate the research hypotheses.

**Table 1. Synthetic software-check results; not competition performance.**

| Method | Brier | Log loss | ECE |
| --- | --- | --- | --- |
| Constant 0.5 | 0.2500 | 0.6931 | 0.0672 |
| Global rate | 0.2464 | 0.6859 | 0.0303 |
| Smoothed system rate | 0.2290 | 0.6500 | 0.0396 |
| Hashed text logistic | 0.2291 | 0.6497 | 0.0516 |
| Metadata logistic | 0.2290 | 0.6500 | 0.0412 |
| Fusion MLP | 0.2116 | 0.6133 | 0.0383 |
| Fusion + temperature | 0.2101 | 0.6086 | 0.0206 |

Figure 1. Synthetic reliability diagrams appear in the PDF. Ten fixed-width bins; empty bins omitted.

The results above confirm that models train and emit finite probabilities under the implemented protocol. The apparent advantage of fusion and temperature scaling is specific to this artificial generator and split. It is not evidence of competition improvement. The saved JSON includes development-selected epochs, temperature, environment, data fingerprint, metrics, reliability counts, and item-cluster intervals; row-level predictions and model weights are also retained.

### 4.2 Verification coverage and gaps
Seven automated checks passed: benchmark/item separation, cross-benchmark duplicate rejection, label validity, metric/probability checks, unknown categories, finite-difference gradients, and learning on a small separable example. These checks establish useful software properties but do not verify a release-specific data join, semantic encoder, retrieval index, adaptation policy, or organizer runtime. Those components cannot be marked complete from this run.

<!-- PAGEBREAK -->
## 5. Progress report and team contributions
The team reported on October 19 that no work had been completed after the original proposal. The repository initially contained the three-page proposal and project website, but no training pipeline or experiment outputs. The original role allocation is retained below as a responsibility plan; it is not evidence that any member completed those tasks. The new artifacts were generated and executed with coding-assistant support during checkpoint preparation.

### 5.1 Deliverable-level progress
| Deliverable | Verified checkpoint status |
| --- | --- |
| Original proposal and website | Present before this session; original PDF verified as 3 pages |
| Data contract and grouped split | Implemented for normalized CSV; tested on synthetic inputs |
| Rate, logistic and fusion baselines | Implemented and executed on synthetic data only |
| Calibration and diagnostics | Scalar temperature, metrics, saved bins and cluster intervals implemented |
| Official data and empirical findings | Not available; account approval and ingestion remain necessary |
| Semantic encoder, retrieval, adaptation | Proposed; not implemented or evaluated |
| Expanded report and web posting | Five-page draft prepared; final team review and live posting pending |

### 5.2 Individual contribution record
| Team member and role | Completed work reported since proposal | Immediate responsibility |
| --- | --- | --- |
| Yi Chen Kuo - Data lead | None reported; confirmation pending | Obtain data, audit joins and splits, reproduce baselines |
| Sudip Manchare - Integration lead | None reported; confirmation pending | Verify reproducibility, assemble release, post and check PDF |
| Jaideep Manupati - Modeling lead | None reported; confirmation pending | Review/run model code, inspect results, explain limitations |
| Akshada Shailesh Rane - Adaptation lead | None reported; confirmation pending | Review leakage/calibration; specify retrieval and acquisition |

Each member should now complete a concrete task, record the action and supporting artifact in CONTRIBUTIONS.md, and replace the corresponding pending entry before submission. The record should distinguish implementing a method, reviewing assistant-generated code, running experiments, and writing analysis. Planned responsibilities must not be restated as past achievements. If no further work is completed, the report must retain that limitation rather than invent a contribution.

### 5.3 Deviations from the proposal and recovery
The original timeline expected ingestion and baselines before October and encoders/fusion by Checkpoint I. The project is behind that schedule. The new prototype supplies a starting implementation, but official-data evaluation remains the critical dependency. The revised priority is to establish a defensible dataset and baseline comparison before adding retrieval or active selection. A smaller, reproducible experiment is preferable to an unsupported claim that the full architecture is operational.

**Assistance disclosure.** Codex assisted with literature/rule verification, code generation, test execution, and this report draft. The measured results are from its executed synthetic run. These activities are not attributed to individual students. Team members remain responsible for understanding the methods, verifying the report, and documenting their actual contributions.

<!-- PAGEBREAK -->
## 6. Completion plan, risks, and next milestones
The immediate course requirement is a five-page updated proposal with a member-level progress report, posted before class on Tuesday, October 20. The class start time is not recorded here, so the operational target is to finish posting on October 19. The following sequence prioritizes a truthful checkpoint over expanding model complexity under deadline pressure.

| Order | Action and acceptance evidence |
| --- | --- |
| 1 - Access and review | Request dataset approval; confirm deadline; every member reads the report |
| 2 - Reproduce | Run seven checks and the synthetic experiment; inspect splits, outputs and limitations |
| 3 - Real data, if available | Pin release, audit/export rows, run same baselines; preserve provenance and results |
| 4 - Contributions | Record actual member tasks and evidence; revise the progress table truthfully |
| 5 - Publish | Rebuild/inspect five pages; upload report and page link; verify public download before class |

For the next research phase, replace hashing with a frozen, versioned sentence encoder and implement learned metadata representations. Compare text-only, metadata-only, and fusion models using the same held-out benchmarks. Add retrieval only after establishing a training-only evidence index. Begin adaptation with random acquisition and test budget accounting before comparing more complex policies. Calibration and ablations must use disjoint selection data; report negative findings as well as improvements.

The main risks are delayed dataset access, leakage across related benchmarks, sparse system coverage, and overfitting from small validation sets. The current dense implementation is suitable for preliminary experiments but will need batching for a large release. Unseen systems require separate stress tests because the synthetic experiment includes every system in training. A saved NumPy checkpoint is a research artifact, not proof of compatibility with the competition interface.

**Conclusion.** ColdCast now has an executable preliminary pipeline and a concrete evaluation plan. Its central scientific question remains open. The next meaningful milestone is a reproducible real-data comparison with honest member contributions and a verified public report link.

## References
[1] Stanford AI Measurement Science. Predictive AI Evaluation Competition, 2026. Current protocol checked October 19, 2026. https://aims.stanford.edu/competition/

[2] S. Truong, Y. Tu, P. Liang, B. Li, and S. Koyejo. Reliable and Efficient Amortized Model-based Evaluation. ICML, 2025. https://arxiv.org/abs/2503.13335

[3] Y. Zhuang et al. Position: AI Evaluation Should Learn from How We Test Humans. ICML, 2025; arXiv:2306.10512v4. https://arxiv.org/abs/2306.10512

[4] H. Jiang, S. Kwon, J. Luo, Z. Xiao, and S. Zhang. Can We Trust Item Response Theory for AI Evaluation? 2026. https://arxiv.org/abs/2607.15190

[5] E. Shang, W. Tang, and Y. He. LLM Evaluation on Unseen Questions: Contextual Multidimensional IRT Model. 2026. https://arxiv.org/abs/2608.22295

[6] AIMS Foundations. measurement-db, dataset card and access conditions. Checked October 19, 2026. https://huggingface.co/datasets/aims-foundations/measurement-db

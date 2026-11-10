"""Replay fixed Checkpoint 1 predictions under three label selection policies."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

CP2 = Path(__file__).resolve().parents[1]
CP1 = CP2.parent / "checkpoint1"
OUT = CP2 / "results"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])
TRIALS = list(range(6200, 6212))
MODELS = ["constant_0.5", "metadata_logistic", "fusion_temperature"]
POLICIES = ["random", "uncertainty", "diversity"]


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def fit_offset(logits, labels):
    """MAP solution of the same unit-Gaussian offset objective used in CP1."""
    if not len(labels):
        return 0.
    def derivative(offset):
        return float(np.sum(expit(logits + offset) - labels) + offset)
    if derivative(-5.) >= 0:
        return -5.
    if derivative(5.) <= 0:
        return 5.
    return brentq(derivative, -5., 5., xtol=1e-10)


def diversify(vectors, order, count):
    """Farthest-first traversal by cosine distance, with paired random tie order."""
    distances = 1 - np.clip(vectors @ vectors.T, -1, 1)
    nearest = np.full(len(order), np.inf)
    available = np.ones(len(order), dtype=bool)
    selected = []
    for _ in range(count):
        candidates = order[available[order]]
        index = candidates[np.argmax(nearest[candidates])]
        selected.append(index)
        available[index] = False
        nearest = np.minimum(nearest, distances[index])
    return np.array(selected)


def acquire(acq_logits, reveal, order, adaptive=False):
    """Return a nested acquisition order and fitted offsets using acquisition data only."""
    offsets = []
    chosen, labels = [], []
    available = np.ones(len(acq_logits), dtype=bool)
    offset = 0.
    for step in range(1, max(BUDGETS) + 1):
        if adaptive:
            candidates = order[available[order]]
            distances = np.abs(acq_logits[candidates] + offset)
            tied = distances <= distances.min() + 1e-7
            index = candidates[np.flatnonzero(tied)[0]]
        else:
            index = order[step - 1]
        available[index] = False
        chosen.append(index)
        labels.append(reveal(index))
        if adaptive or step in BUDGETS:
            offset = fit_offset(acq_logits[chosen], np.array(labels))
        if step in BUDGETS:
            offsets.append(offset)
    return offsets, chosen


def evaluate(acq_logits, reveal, evaluation_logits, evaluation_labels, evaluation_base, order, adaptive=False):
    offsets, chosen = acquire(acq_logits, reveal, order, adaptive)
    predictions = [evaluation_base, *(expit(evaluation_logits + offset) for offset in offsets)]
    scores = [np.mean((p - evaluation_labels)**2) for p in predictions]
    return np.array(scores), chosen


def original_pools(frame, source, system):
    keys = np.array(sorted(frame.text_group.unique()))
    rng = np.random.default_rng(seed_for(source, system))
    rng.shuffle(keys)
    groups = keys[:len(keys)//2]
    # Reconstruct CP1's exact row choices. Integer positions preserve input row order.
    positions = frame.groupby("text_group", sort=False).indices
    chosen = [positions[key][rng.integers(len(positions[key]))] for key in groups]
    evaluation = frame[~frame.text_group.isin(groups)]
    assert set(groups).isdisjoint(evaluation.text_group)
    return frame.iloc[chosen], evaluation


def legacy_scores(acquisition, evaluation, model):
    """Reproduce CP1's bounded minimizer and stored prediction dtypes."""
    base = evaluation[model]
    y_eval = evaluation.label.to_numpy()
    scores = [float(np.mean((base.to_numpy() - y_eval)**2))]
    for budget in BUDGETS[1:]:
        z = logit(np.clip(acquisition[model].iloc[:budget], 1e-6, 1 - 1e-6))
        y = acquisition.label.to_numpy()[:budget]

        def objective(offset):
            return np.sum(np.logaddexp(0, z + offset) - y * (z + offset)) + .5 * offset**2

        offset = minimize_scalar(objective, bounds=(-5, 5), method="bounded").x
        probabilities = expit(logit(np.clip(base, 1e-6, 1 - 1e-6)) + offset)
        scores.append(float(np.mean((np.asarray(probabilities) - y_eval)**2)))
    return np.array(scores)


def log_odds(probabilities):
    return logit(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6))


def aggregate(results):
    keys = ["model", "policy", "budget"]
    curves = results.groupby(keys, as_index=False).brier.mean()
    curves.to_csv(OUT / "learning_curves.csv", index=False)
    pair_keys = ["source", "system", "seed", "trial", "model", "policy"]
    alcs = results.pivot(index=pair_keys, columns="budget", values="brier").reset_index()
    alcs["alc"] = alcs[BUDGETS].to_numpy() @ WEIGHTS
    trial = alcs.groupby(["model", "policy", "trial"], as_index=False).alc.mean()
    trial.to_csv(OUT / "trial_alc.csv", index=False)
    stats = trial.groupby(["model", "policy"]).alc.agg(alc="mean", trial_sd="std").reset_index()
    stats.to_csv(OUT / "policy_summary.csv", index=False)
    by_source = alcs.groupby(["model", "source", "policy"], as_index=False).alc.mean()
    by_source.to_csv(OUT / "source_alc.csv", index=False)
    table = by_source.pivot(index=["model", "source"], columns="policy", values="alc")
    records = []
    for model, frame in table.groupby(level="model"):
        frame = frame.droplevel("model")
        for policy in POLICIES[1:]:
            delta = frame[policy] - frame.random
            loo = [(delta.sum() - value)/(len(delta)-1) for value in delta]
            records.append(dict(model=model, policy=policy, equal_source_delta=delta.mean(),
                                leave_one_source_out_min=min(loo), leave_one_source_out_max=max(loo)))
    pd.DataFrame(records).to_csv(OUT / "source_sensitivity.csv", index=False)


def main():
    start = perf_counter()
    OUT.mkdir(exist_ok=True)
    real = CP1 / "results/real"
    inputs = {
        "held_out_predictions": real / "test_predictions.parquet",
        "pool_manifest": real / "adaptation_pools.json",
        "reference_adaptation_scores": real / "adaptation_by_pair.csv",
        "frozen_embeddings": CP1 / "data/real_embeddings.npz",
    }
    missing = [str(path.relative_to(CP1.parent)) for path in inputs.values() if not path.is_file()]
    if missing:
        raise SystemExit("Regenerate the baseline artifacts using checkpoint1/REAL_RESULTS.md. Missing: "
                         + ", ".join(missing))
    input_hashes = {}
    for name, path in inputs.items():
        with path.open("rb") as handle:
            input_hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    predictions = pd.read_parquet(inputs["held_out_predictions"])
    saved_pools = {(row["source"], row["system_key"]): row
                   for row in json.loads(inputs["pool_manifest"].read_text())}
    baseline = pd.read_csv(inputs["reference_adaptation_scores"])
    baseline = baseline.set_index(["seed", "source", "system_key", "model", "budget"]).brier.sort_index()
    with np.load(inputs["frozen_embeddings"]) as data:
        embeddings = dict(zip(data["keys"], data["vectors"]))
    held = predictions[predictions.protocol == "benchmark"]
    rows = []
    pair_counts = {}
    match_count = flat_count = fusion_count = 0
    max_replay_error = max_solver_change = 0.
    for (source, system), pair in held.groupby(["source", "system_key"], sort=True):
        if (source, system) not in saved_pools:
            assert pair.text_group.nunique() < 80
            continue
        pair_counts[source] = pair_counts.get(source, 0) + 1
        pool = saved_pools[(source, system)]
        by_seed = list(pair.groupby("seed", sort=True))
        groups = pool["acquisition_groups"]
        vectors = np.array([embeddings[key] for key in groups])
        orders = [np.random.default_rng(seed_for(source, system, trial)).permutation(len(groups))
                  for trial in TRIALS]
        diversity_orders = [diversify(vectors, order, max(BUDGETS)) for order in orders]
        for training_seed, frame in by_seed:
            acq, evaluation = original_pools(frame, source, system)
            assert acq.text_group.tolist() == groups
            assert sorted(evaluation.text_group.unique()) == pool["evaluation_groups"]
            labels = acq.label.to_numpy()
            evaluation_labels = evaluation.label.to_numpy()
            for model in MODELS:
                eval_base = evaluation[model].to_numpy(dtype=float)
                a = log_odds(acq[model])
                e = log_odds(eval_base)
                replay, _ = evaluate(a, labels.__getitem__, e, evaluation_labels, eval_base,
                                     np.arange(len(a)))
                expected = baseline.loc[(training_seed, source, system, model)].reindex(BUDGETS).to_numpy()
                legacy = legacy_scores(acq, evaluation, model)
                max_replay_error = max(max_replay_error, np.max(np.abs(legacy - expected)))
                max_solver_change = max(max_solver_change, np.max(np.abs(replay - legacy)))
                if model == "fusion_temperature":
                    fusion_count += 1
                elif np.ptp(a) <= 1e-7:
                    flat_count += 1
                for trial, random_order, diverse_order in zip(TRIALS, orders, diversity_orders):
                    for policy, order in zip(POLICIES, [random_order, random_order, diverse_order]):
                        scores, chosen = evaluate(a, labels.__getitem__, e, evaluation_labels,
                                                  eval_base, order, policy == "uncertainty")
                        assert len(chosen) == len(set(chosen)) == max(BUDGETS)
                        if policy == "uncertainty" and model == "fusion_temperature":
                            match_count += int(chosen == random_order[:max(BUDGETS)].tolist())
                        rows.extend((source, system, training_seed, trial, model, policy, b, score)
                                    for b, score in zip(BUDGETS, scores))
    assert sum(pair_counts.values()) == len(saved_pools) == 122
    assert max_replay_error < 1e-10, max_replay_error
    assert max_solver_change < 1e-4, max_solver_change
    results = pd.DataFrame(rows, columns=["source", "system", "seed", "trial", "model", "policy", "budget", "brier"])
    expected_rows = 122 * 3 * len(TRIALS) * len(MODELS) * len(POLICIES) * len(BUDGETS)
    assert len(results) == expected_rows and results.brier.between(0, 1).all()
    assert results[results.budget == 0].groupby(["source", "system", "seed", "model"]).brier.nunique().eq(1).all()
    aggregate(results)
    details = dict(budgets=BUDGETS, weights=WEIGHTS.tolist(), trials=TRIALS,
                   training_seeds=sorted(held.seed.unique().tolist()), pairs=sum(pair_counts.values()),
                   source_pairs=pair_counts, score_count=len(results), input_sha256=input_hashes,
                   max_cp1_replay_error=float(max_replay_error), max_solver_change=float(max_solver_change),
                   flat_non_neural_pair_seeds=flat_count,
                   identical_fusion_uncertainty_random_orders=match_count,
                   fusion_order_comparisons=fusion_count * len(TRIALS),
                   seconds=round(perf_counter() - start, 3))
    (OUT / "run.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))
    print(pd.read_csv(OUT / "policy_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()

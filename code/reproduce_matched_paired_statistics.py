#!/usr/bin/env python3
"""Offline, read-only replay of exploratory 198/132-case paired tests.

Run from the repository root with: python code/reproduce_matched_paired_statistics.py
Requires numpy==2.3.5 (Python 3.11+); never calls network or live models.
Nothing in this script writes to the source data, results, or filesystem.
"""
from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "matched_nonsingular_198_case_outcomes.csv"
EXPECTED = ROOT / "results" / "non_singular_paired_statistics.csv"
COHORT = ROOT / "subsets" / "matched_198.csv"
SOURCE_SHA256 = "214f63d7d5711538220a21a46dc5ebdb9a0ae148276ded1d58e8be0ab0aa0325"
MODELS = ("qwen", "gemma", "gemini", "gpt")
COMPARISONS = (
    ("Text only vs Direct", "direct", "text"),
    ("Assisted vs Direct", "direct", "assisted"),
    ("Assisted vs Text only", "text", "assisted"),
)
REPS = 5000
SEED = 20261016


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing expected file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def exact_two_sided_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    smaller = min(gains, losses)
    # The exact conditional two-sided binomial test at probability 0.5.
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(smaller + 1)) / 2 ** n)


def bootstrap_ci(case_rows: list[dict[str, str]], before: str, after: str) -> tuple[float, float]:
    # Match source pandas groupby(sort=True): group keys sorted ascending;
    # within-stratum rows retain their order in the original CSV.
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(case_rows):
        groups[(row["mechanism_family"], row["benchmark_state"])].append(i)
    strata = [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
    delta = np.array([
        int(row[after + "_physical_exact_success"]) - int(row[before + "_physical_exact_success"])
        for row in case_rows
    ], dtype=np.int8)
    rng = np.random.default_rng(SEED)
    simulations = np.empty(REPS)
    for i in range(REPS):
        sample_indices = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in strata
        ])
        simulations[i] = 100 * delta[sample_indices].mean()
    return tuple(round(float(x), 4) for x in np.quantile(simulations, (0.025, 0.975)))


def computed_rows(input_rows: list[dict[str, str]]) -> list[dict]:
    findings = []
    for model in MODELS:
        model_rows = [r for r in input_rows if r["model"] == model]
        require(len(model_rows) == 198, f"Wrong number of matched cases for {model}")
        for cohort, records in (
            ("matched_non_singular_198", model_rows),
            ("matched_regular_near_132", [
                r for r in model_rows if r["benchmark_state"] in ("regular", "near_singular")
            ]),
        ):
            n = 198 if cohort.endswith("198") else 132
            require(len(records) == n, f"Wrong {cohort} count for {model}")
            for comparison, before, after in COMPARISONS:
                a = np.asarray([int(r[before + "_physical_exact_success"]) for r in records], dtype=np.int8)
                b = np.asarray([int(r[after + "_physical_exact_success"]) for r in records], dtype=np.int8)
                gains = int(np.count_nonzero((a == 0) & (b == 1)))
                losses = int(np.count_nonzero((a == 1) & (b == 0)))
                lower, upper = bootstrap_ci(records, before, after)
                findings.append({
                    "model": model, "cohort": cohort, "comparison": comparison,
                    "n": n, "reference_successes": int(a.sum()), "comparison_successes": int(b.sum()),
                    "delta_pp": round(float(100 * (b-a).mean()), 4),
                    "gains": gains, "losses": losses,
                    "McNemar_exact_p_exploratory": exact_two_sided_mcnemar(gains, losses),
                    "CI95_stratified_paired_bootstrap_lower_pp": lower,
                    "CI95_stratified_paired_bootstrap_upper_pp": upper,
                })
    for cohort in ("matched_non_singular_198", "matched_regular_near_132"):
        for comparison, _, _ in COMPARISONS:
            matched = [f for f in findings if f["cohort"] == cohort and f["comparison"] == comparison]
            ranked = sorted(matched, key=lambda f: f["McNemar_exact_p_exploratory"])
            running = 0.0
            for i, f in enumerate(ranked):
                running = max(running, (4-i) * f["McNemar_exact_p_exploratory"])
                f["Holm_p_across_4_models_posthoc"] = min(1.0, running)
    return findings


def main() -> None:
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA256,
            "The archived per-case outcomes do not match the source SHA-256")
    cases = read_csv(SOURCE)
    require(len(cases) == 792, "Expect 4 models x 198 archived matched cases")
    cohort = read_csv(COHORT)
    case_ids = {r["case_uid"] for r in cohort}
    require(len(case_ids) == len(cohort) == 198, "Invalid 198-case subset identifier list")
    require(set(r["benchmark_state"] for r in cases) == {"regular", "near_singular", "infeasible"},
            "Historical singular cases were included unexpectedly")
    require(Counter(r["model"] for r in cases) == {m:198 for m in MODELS},
            "Unequal per-model case numbers")
    for model in MODELS:
        r_model = [r for r in cases if r["model"] == model]
        require({r["case_uid"] for r in r_model} == case_ids,
                f"Matched cases disagree with release subset for {model}")
        require(Counter((r["mechanism_family"], r["benchmark_state"]) for r in r_model)
                == {(f,s):22 for f in ("RRRR", "RRRP", "5R") for s in ("regular", "near_singular", "infeasible")},
                "Incorrect nine-stratum balance")
        for r in r_model:
            for c in ("direct", "text", "assisted"):
                require(r[c+"_physical_exact_success"] in ("0", "1"), "Nonbinary physical outcome")
    recorded = read_csv(EXPECTED)
    computed = computed_rows(cases)
    require(len(recorded) == len(computed) == 24, "Expected exactly 24 paired comparisons")
    expected_map = {(r["model"],r["cohort"],r["comparison"]):r for r in recorded}
    require(len(expected_map)==24, "Duplicate archived statistics rows")
    int_fields = ("n", "reference_successes", "comparison_successes", "gains", "losses")
    float_fields = ("delta_pp", "McNemar_exact_p_exploratory",
                    "CI95_stratified_paired_bootstrap_lower_pp",
                    "CI95_stratified_paired_bootstrap_upper_pp", "Holm_p_across_4_models_posthoc")
    for row in computed:
        key=(row["model"],row["cohort"],row["comparison"])
        require(key in expected_map, f"Missing comparison: {key}")
        reference=expected_map[key]
        for col in int_fields:
            require(int(reference[col]) == row[col], f"Integer discrepancy for {key}, {col}")
        for col in float_fields:
            require(math.isclose(float(reference[col]), row[col], rel_tol=1e-10, abs_tol=5e-5),
                    f"Float discrepancy for {key}, {col}: archived={reference[col]}, computed={row[col]}")
    print("PASS: 792 source model-case rows; 198 matched cases/model; 132 feasible cases/model.")
    print("PASS: All 24 paired comparisons agree with archived exact McNemar p-values, bootstrap CIs, and Holm p-values.")
    print("No model requests and no source file changes.")


if __name__ == "__main__":
    main()

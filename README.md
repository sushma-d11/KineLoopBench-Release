# KineLoopBench

Benchmark v1.0 candidate: 225 primary prompt-consistent mechanism cases across RRRR, RRRP and planar 5R, with 150 feasible and 198 matched subsets. 75 nominal exact-singular cases are retained separately as historical records because rounded prompts have changed solution cardinality (43 zero, 32 two).

The frozen `code/scoring_core_frozen_v1.py` uses exact configuration count, order-independent branch matching, 1 mm position, 1 degree angle and 0.01 normalized residual tolerances.

Use `python verify_dataset.py`, `python code/reproduce_primary_scores.py`, and `python code/reproduce_matched_paired_statistics.py` to reproduce data integrity, saved score totals and statistical comparisons. Install `requirements-analysis.txt` first. Raw model responses are not included, so response-text rescoring and hosted inference cannot be reproduced from this package alone.

Original research files are unchanged. Generator scripts are included for provenance, not for execution. Software: MIT. Original data and schematics: CC BY 4.0. Creator: Sushma Das. Manuscript not yet published.

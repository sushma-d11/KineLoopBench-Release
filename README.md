# KineLoopBench

**Public benchmark release: v1.0.0** · [Repository](https://github.com/sushma-d11/KineLoopBench-Release)

KineLoopBench evaluates whether multimodal models can reconstruct the **complete set of physically admissible assembly configurations** for three planar closed-loop mechanism families, instead of only predicting a plausible answer or the correct number of configurations.

The three families are **RRRR four-bar**, **offset RRRP slider-crank**, and **planar 5R parallel linkage**. The benchmark compares three conditions: **Direct Image + Text**, **Text Only**, and **Image + Text + Exact Count** (the last receives the correct number of configurations but no reference coordinates or solution geometry).

## Dataset and benchmark scope

| Location | Contents |
| --- | --- |
| `data/primary_225/` | **225 input-consistent primary cases**: 25 regular, 25 near-singular and 25 infeasible cases **per mechanism family** |
| `subsets/feasible_150.csv` | **150 feasible primary cases** with two assembly configurations |
| `subsets/matched_198.csv` | **198 primary cases** matched across all three input conditions |
| `data/historical_75_excluded/` | **75 original nominal exact-singular cases**, preserved **only as historical/exploratory material** |
| `assets/model_input_schematics/` | The three fixed family-level input schematics |
| `code/` | Three analytical reference solvers, original frozen scorer, offline verification scripts and original generator source |
| `protocol/` | Model response instructions |
| `results/` | Derived evaluation outcomes and saved statistical comparisons |
| `verify_dataset.py` | Dataset membership, checksum and structural integrity verification |

**The 150 feasible and 198 matched cases overlap with the 225 primary cases; they must not be added together.** The primary benchmark excludes the original 75 nominal exact-singular cases after a post-collection precision audit: rounding the model-facing inputs changed their intended single tangent solution to **zero real intersections in 43 cases** and **two intersections in 32 cases**. Original historical files and hashes have not been modified. Post-collection comparisons are exploratory and are not claimed as prespecified confirmatory tests.

## Model-facing input and expected output

The fixed mechanism-family schematic provides topology and variable labels. **All case-specific numerical parameters are supplied in text.** The image comparison therefore tests the incremental value of the fixed schematic, not dimension extraction from previously unseen engineering drawings.

Each model should return an unwrapped JSON object with an integer `configuration_count` and a list `configurations`. Expected configuration fields are:

- **RRRR and 5R:** `xC_mm`, `yC_mm`, `theta3_deg`, `theta4_deg`
- **RRRP:** `xC_mm`, `theta3_deg`
- **Infeasible:** `"configuration_count": 0` and `"configurations": []`

The frozen scorer `code/scoring_core_frozen_v1.py` evaluates correct cardinality, complete **order-independent branch matching**, position error **≤1 mm**, circular angle error **≤1°**, and normalized physical closure residual **≤0.01**. A correct count is *not* equivalent to complete physical-exact success.

## Reproduce the offline results

Install Python 3.11 or newer and run the following commands from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python verify_dataset.py
python code/reproduce_primary_scores.py
python code/reproduce_matched_paired_statistics.py
```

The three checks passed both on the prepared package and in an independent Windows Python 3.13 virtual environment with pinned **NumPy 2.3.5**. They validate the integrity of the **300 original cases** (225 primary + 75 historical), the cohort definitions, all **32 aggregate primary/feasible score totals**, and all **24 paired statistical comparisons**.

**Reproducibility limitation:** The repository includes *derived model-scoring outcomes and statistics*, but **not the original archived model-response texts or hosted inference records**. Consequently, readers can reproduce the released offline aggregations and paired analyses, but **cannot independently rerun original hosted inference or rescore the original raw responses from this package alone**. Historical model/API runner scripts are intentionally excluded. The original generators in `code/archived_generators/` are provided for provenance and may modify files or rely on legacy paths; **do not run these scripts as turnkey reproduction commands**.

## License and citation

Original Python software: **MIT** ([LICENSE](LICENSE)). Original benchmark cases, schematics, results and other non-software material: **CC BY 4.0** ([LICENSE-DATA.md](LICENSE-DATA.md)).

Creator: **Sushma Das**. Citation information is available in [CITATION.cff](CITATION.cff). The associated research manuscript, *KineLoopBench: Solver-Grounded Evaluation of Multimodal Models for Complete Configuration Recovery in Planar Mechanisms*, is being prepared for journal submission; **no published article DOI or Zenodo DOI is claimed**.

This is an independent public release with new Git history. The original private development repository and its historical commits remain separate.

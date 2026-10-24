import os
import json
import shutil
import random
import math

from rrrp_slider_crank_solver import solve_slider_crank


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "RRRP_Auto_Test"
)

FINAL_RRRP_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "KineBench_RRRP"
)

CANONICAL_IMAGE = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "Pilot_RRRP",
    "sample_0001",
    "image.png"
)

print("Output folder:", OUTPUT_ROOT)
print("Canonical image:", CANONICAL_IMAGE)

# ---------------------------------------------------------
# Benchmark state threshold
# ---------------------------------------------------------

NEAR_SINGULAR_RHO = 0.05

# ---------------------------------------------------------
# Full RRRP benchmark sampling settings
# ---------------------------------------------------------

RANDOM_SEED = 2026
EXPORT_SHUFFLE_SEED = 2027

N_PER_STATE = 25

TARGET_COUNTS = {
    "regular": N_PER_STATE,
    "near_singular": N_PER_STATE,
    "singular": N_PER_STATE,
    "infeasible": N_PER_STATE
}

accepted_cases = {
    "regular": [],
    "near_singular": [],
    "singular": [],
    "infeasible": []
}

L3_FIXED = 100.0

L2_RATIO_MIN = 0.30
L2_RATIO_MAX = 1.20

OFFSET_RATIO_MIN = -1.50
OFFSET_RATIO_MAX = 1.50

THETA2_MIN = 0.0
THETA2_MAX = 360.0

rng = random.Random(RANDOM_SEED)


def classify_rrrp_state(result):

    if result["feasible"] is False:
        return "infeasible"

    if result["status"] == "tangent":
        return "singular"

    if result["n_branches"] == 2:

        rho_values = [
            solution["singularity_score"]
            for solution in result["solutions"]
        ]

        min_rho = min(rho_values)

        if min_rho <= NEAR_SINGULAR_RHO:
            return "near_singular"

        return "regular"

    return "unclassified"


# ---------------------------------------------------------
# Verified pilot test cases
# ---------------------------------------------------------

CASES = [
    {
        "sample_id": "sample_0001",
        "L2": 40.0,
        "L3": 70.0,
        "e": 20.0,
        "theta2_deg": 60.0
    },
    {
        "sample_id": "sample_0002",
        "L2": 40.0,
        "L3": 70.0,
        "e": 70.0,
        "theta2_deg": 0.1
    },
    {
        "sample_id": "sample_0003",
        "L2": 40.0,
        "L3": 70.0,
        "e": 70.0,
        "theta2_deg": 0.0
    },
    {
        "sample_id": "sample_0004",
        "L2": 40.0,
        "L3": 70.0,
        "e": 80.0,
        "theta2_deg": 0.0
    }
]

# ---------------------------------------------------------
# Automatically package the verified pilot cases
# ---------------------------------------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for case in CASES:

    sample_id = case["sample_id"]

    L2 = case["L2"]
    L3 = case["L3"]
    e = case["e"]
    theta2_deg = case["theta2_deg"]

    # Create sample folder
    sample_dir = os.path.join(
        OUTPUT_ROOT,
        sample_id
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    # Exact ground truth
    result = solve_slider_crank(
        L2=L2,
        L3=L3,
        e=e,
        theta2_deg=theta2_deg
    )

    benchmark_state = classify_rrrp_state(result)

    ground_truth = dict(result)

    ground_truth["benchmark_state"] = benchmark_state
    ground_truth["near_singular_rho_threshold"] = (
        NEAR_SINGULAR_RHO
    )

    ground_truth_path = os.path.join(
        sample_dir,
        "ground_truth.json"
    )

    with open(ground_truth_path, "w") as f:
        json.dump(
            ground_truth,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt_text = f"""The image shows a planar offset RRRP slider-crank mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L2 = {L2:g} mm
L3 = {L3:g} mm
e = {e:g} mm
theta2 = {theta2_deg:g} degrees

Angle convention:
theta2 is measured counter-clockwise from the positive x-axis.
theta3 is the orientation of connecting rod BC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Slider convention:
The slider moves along the horizontal line y = e.
The slider position is C = (xC, e).

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. xC and theta3 for every valid configuration.

If no physically valid configuration exists, report zero configurations.
"""

    prompt_path = os.path.join(
        sample_dir,
        "prompt.txt"
    )

    with open(prompt_path, "w") as f:
        f.write(prompt_text)

    # -----------------------------------------------------
    # Canonical image
    # -----------------------------------------------------

    image_output_path = os.path.join(
        sample_dir,
        "image.png"
    )

    shutil.copy2(
        CANONICAL_IMAGE,
        image_output_path
    )

    print(
        sample_id,
        "| solver status =", result["status"],
        "| benchmark state =", benchmark_state,
        "| configurations =", result["n_branches"]
    )


print("RRRP automatic test packaging complete.")

def generate_random_rrrp_candidate():

    L3 = L3_FIXED

    L2 = L3 * rng.uniform(
        L2_RATIO_MIN,
        L2_RATIO_MAX
    )

    e = L3 * rng.uniform(
        OFFSET_RATIO_MIN,
        OFFSET_RATIO_MAX
    )

    theta2_deg = rng.uniform(
        THETA2_MIN,
        THETA2_MAX
    )

    return {
        "L2": L2,
        "L3": L3,
        "e": e,
        "theta2_deg": theta2_deg
    }

def collect_random_cases_for_state(target_state, n_needed):

    collected = []
    attempts = 0

    max_attempts = 100000

    while len(collected) < n_needed:

        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not collect {n_needed} cases for state "
                f"'{target_state}' within {max_attempts} attempts."
            )

        candidate = generate_random_rrrp_candidate()

        result = solve_slider_crank(
            **candidate
        )

        benchmark_state = classify_rrrp_state(result)

        attempts += 1

        if benchmark_state == target_state:
            collected.append(candidate)

    return collected, attempts

def generate_exact_singular_candidate():

    while True:

        L3 = L3_FIXED

        L2 = L3 * rng.uniform(
            L2_RATIO_MIN,
            L2_RATIO_MAX
        )

        theta2_deg = rng.uniform(
            THETA2_MIN,
            THETA2_MAX
        )

        theta2 = math.radians(theta2_deg)

        By = L2 * math.sin(theta2)

        # Choose upper or lower tangent condition
        if rng.random() < 0.5:
            tangent_sign = +1
        else:
            tangent_sign = -1

        e = By + tangent_sign * L3

        # Keep offset inside the benchmark range
        e_min = L3 * OFFSET_RATIO_MIN
        e_max = L3 * OFFSET_RATIO_MAX

        if not (e_min <= e <= e_max):
            continue

        candidate = {
            "L2": L2,
            "L3": L3,
            "e": e,
            "theta2_deg": theta2_deg
        }

        result = solve_slider_crank(
            **candidate
        )

        # Keep only solver-confirmed exact singular cases
        if classify_rrrp_state(result) == "singular":
            return candidate


def generate_near_singular_candidate():

    while True:

        L3 = L3_FIXED

        L2 = L3 * rng.uniform(
            L2_RATIO_MIN,
            L2_RATIO_MAX
        )

        theta2_deg = rng.uniform(
            THETA2_MIN,
            THETA2_MAX
        )

        theta2 = math.radians(theta2_deg)

        By = L2 * math.sin(theta2)

        # Choose a small positive singularity score
        target_rho = rng.uniform(
            0.005,
            NEAR_SINGULAR_RHO
        )

        # From rho = q / L3
        vertical_magnitude = (
            L3
            * math.sqrt(
                1.0 - target_rho**2
            )
        )

        # Upper or lower near-tangent branch
        if rng.random() < 0.5:
            vertical_sign = +1
        else:
            vertical_sign = -1

        e = By + vertical_sign * vertical_magnitude

        # Keep offset inside benchmark range
        e_min = L3 * OFFSET_RATIO_MIN
        e_max = L3 * OFFSET_RATIO_MAX

        if not (e_min <= e <= e_max):
            continue

        candidate = {
            "L2": L2,
            "L3": L3,
            "e": e,
            "theta2_deg": theta2_deg
        }

        result = solve_slider_crank(
            **candidate
        )

        # Keep only solver-confirmed near-singular cases
        if classify_rrrp_state(result) == "near_singular":
            return candidate

def build_balanced_rrrp_case_pool():

    # Regular cases
    regular_cases, regular_attempts = (
        collect_random_cases_for_state(
            "regular",
            N_PER_STATE
        )
    )

    # Infeasible cases
    infeasible_cases, infeasible_attempts = (
        collect_random_cases_for_state(
            "infeasible",
            N_PER_STATE
        )
    )

    # Exact singular cases
    singular_cases = [
        generate_exact_singular_candidate()
        for _ in range(N_PER_STATE)
    ]

    # Near-singular cases
    near_singular_cases = [
        generate_near_singular_candidate()
        for _ in range(N_PER_STATE)
    ]

    accepted_cases["regular"] = regular_cases
    accepted_cases["near_singular"] = near_singular_cases
    accepted_cases["singular"] = singular_cases
    accepted_cases["infeasible"] = infeasible_cases

    statistics = {
        "regular_attempts": regular_attempts,
        "infeasible_attempts": infeasible_attempts
    }

    return accepted_cases, statistics

def audit_rrrp_case_pool(case_pool):

    expected_branches = {
        "regular": 2,
        "near_singular": 2,
        "singular": 1,
        "infeasible": 0
    }

    total_cases = 0
    state_errors = []
    branch_errors = []
    validation_errors = []
    duplicate_cases = []

    seen_parameters = set()

    for intended_state, cases in case_pool.items():

        for index, candidate in enumerate(cases, start=1):

            total_cases += 1

            result = solve_slider_crank(
                **candidate
            )

            actual_state = classify_rrrp_state(result)

            # Check state
            if actual_state != intended_state:
                state_errors.append(
                    (
                        intended_state,
                        index,
                        actual_state
                    )
                )

            # Check branch count
            if result["n_branches"] != expected_branches[intended_state]:
                branch_errors.append(
                    (
                        intended_state,
                        index,
                        result["n_branches"]
                    )
                )

            # Check geometric validation
            for solution in result["solutions"]:

                if not solution["validation_passed"]:
                    validation_errors.append(
                        (
                            intended_state,
                            index
                        )
                    )

            # Check duplicate parameter sets
            parameter_key = (
                round(candidate["L2"], 10),
                round(candidate["L3"], 10),
                round(candidate["e"], 10),
                round(candidate["theta2_deg"], 10)
            )

            if parameter_key in seen_parameters:
                duplicate_cases.append(
                    (
                        intended_state,
                        index
                    )
                )

            seen_parameters.add(parameter_key)

    return {
        "total_cases": total_cases,
        "state_errors": state_errors,
        "branch_errors": branch_errors,
        "validation_errors": validation_errors,
        "duplicate_cases": duplicate_cases
    }

def export_rrrp_case_pool(case_pool):

    os.makedirs(FINAL_RRRP_ROOT, exist_ok=True)

    # Flatten all state groups
    export_cases = []

    for benchmark_state, cases in case_pool.items():

        for candidate in cases:

            export_cases.append(
                {
                    "benchmark_state": benchmark_state,
                    "candidate": candidate
                }
            )

    # Shuffle reproducibly so sample ID does not reveal state
    export_rng = random.Random(EXPORT_SHUFFLE_SEED)
    export_rng.shuffle(export_cases)

    for sample_number, item in enumerate(
        export_cases,
        start=1
    ):

        sample_id = f"sample_{sample_number:04d}"

        benchmark_state = item["benchmark_state"]
        candidate = item["candidate"]

        L2 = candidate["L2"]
        L3 = candidate["L3"]
        e = candidate["e"]
        theta2_deg = candidate["theta2_deg"]

        # Re-solve during export
        result = solve_slider_crank(
            **candidate
        )

        verified_state = classify_rrrp_state(result)

        if verified_state != benchmark_state:
            raise RuntimeError(
                f"{sample_id}: expected '{benchmark_state}' "
                f"but solver returned '{verified_state}'."
            )

        # Create folder
        sample_dir = os.path.join(
            FINAL_RRRP_ROOT,
            sample_id
        )

        os.makedirs(
            sample_dir,
            exist_ok=True
        )

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        ground_truth = dict(result)

        ground_truth["sample_id"] = sample_id
        ground_truth["mechanism_family"] = "RRRP"

        ground_truth["input_parameters"] = {
            "L2_mm": L2,
            "L3_mm": L3,
            "e_mm": e,
            "theta2_deg": theta2_deg
        }

        ground_truth["benchmark_state"] = benchmark_state
        ground_truth["near_singular_rho_threshold"] = (
            NEAR_SINGULAR_RHO
        )

        ground_truth_path = os.path.join(
            sample_dir,
            "ground_truth.json"
        )

        with open(ground_truth_path, "w") as f:
            json.dump(
                ground_truth,
                f,
                indent=4
            )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt_text = f"""The image shows a planar offset RRRP slider-crank mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L2 = {L2:g} mm
L3 = {L3:g} mm
e = {e:g} mm
theta2 = {theta2_deg:g} degrees

Angle convention:
theta2 is measured counter-clockwise from the positive x-axis.
theta3 is the orientation of connecting rod BC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Slider convention:
The slider moves along the horizontal line y = e.
The slider position is C = (xC, e).

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. xC and theta3 for every valid configuration.

If no physically valid configuration exists, report zero configurations.
"""

        prompt_path = os.path.join(
            sample_dir,
            "prompt.txt"
        )

        with open(prompt_path, "w") as f:
            f.write(prompt_text)

        # -------------------------------------------------
        # Canonical image
        # -------------------------------------------------

        image_output_path = os.path.join(
            sample_dir,
            "image.png"
        )

        shutil.copy2(
            CANONICAL_IMAGE,
            image_output_path
        )

    print(
        "Final RRRP samples exported:",
        len(export_cases)
    )

def audit_exported_rrrp_dataset():

    expected_files = {
        "ground_truth.json",
        "prompt.txt",
        "image.png"
    }

    expected_state_counts = {
        "regular": 25,
        "near_singular": 25,
        "singular": 25,
        "infeasible": 25
    }

    sample_folders = sorted(
        name
        for name in os.listdir(FINAL_RRRP_ROOT)
        if name.startswith("sample_")
        and os.path.isdir(
            os.path.join(FINAL_RRRP_ROOT, name)
        )
    )

    missing_files = []
    sample_id_errors = []
    mechanism_errors = []
    state_errors = []
    branch_errors = []
    validation_errors = []
    duplicate_cases = []

    state_counts = {
        "regular": 0,
        "near_singular": 0,
        "singular": 0,
        "infeasible": 0
    }

    seen_parameters = set()

    for sample_folder in sample_folders:

        sample_dir = os.path.join(
            FINAL_RRRP_ROOT,
            sample_folder
        )

        existing_files = set(
            os.listdir(sample_dir)
        )

        missing = expected_files - existing_files

        if missing:
            missing_files.append(
                (
                    sample_folder,
                    sorted(missing)
                )
            )
            continue

        ground_truth_path = os.path.join(
            sample_dir,
            "ground_truth.json"
        )

        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        if ground_truth["sample_id"] != sample_folder:
            sample_id_errors.append(
                sample_folder
            )

        if ground_truth["mechanism_family"] != "RRRP":
            mechanism_errors.append(
                sample_folder
            )

        inputs = ground_truth["input_parameters"]

        candidate = {
            "L2": inputs["L2_mm"],
            "L3": inputs["L3_mm"],
            "e": inputs["e_mm"],
            "theta2_deg": inputs["theta2_deg"]
        }

        intended_state = ground_truth["benchmark_state"]

        result = solve_slider_crank(
            **candidate
        )

        actual_state = classify_rrrp_state(
            result
        )

        if actual_state != intended_state:
            state_errors.append(
                (
                    sample_folder,
                    intended_state,
                    actual_state
                )
            )

        if intended_state in state_counts:
            state_counts[intended_state] += 1

        expected_branches = {
            "regular": 2,
            "near_singular": 2,
            "singular": 1,
            "infeasible": 0
        }

        if (
            result["n_branches"]
            != expected_branches[intended_state]
        ):
            branch_errors.append(
                sample_folder
            )

        for solution in result["solutions"]:

            if not solution["validation_passed"]:
                validation_errors.append(
                    sample_folder
                )

        parameter_key = (
            round(candidate["L2"], 10),
            round(candidate["L3"], 10),
            round(candidate["e"], 10),
            round(candidate["theta2_deg"], 10)
        )

        if parameter_key in seen_parameters:
            duplicate_cases.append(
                sample_folder
            )

        seen_parameters.add(parameter_key)

    report = {
        "folder_count": len(sample_folders),
        "state_counts": state_counts,
        "expected_state_counts": expected_state_counts,
        "missing_files": missing_files,
        "sample_id_errors": sample_id_errors,
        "mechanism_errors": mechanism_errors,
        "state_errors": state_errors,
        "branch_errors": branch_errors,
        "validation_errors": validation_errors,
        "duplicate_cases": duplicate_cases
    }

    return report

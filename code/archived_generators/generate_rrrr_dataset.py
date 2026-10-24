import os
import json
import shutil
import random
import math

from rrrr_fourbar_solver import solve_fourbar

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "RRRR_Auto_Test"
)

FINAL_RRRR_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "KineBench_RRRR"
)


CANONICAL_IMAGE = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "Pilot_RRRR",
    "sample_0001",
    "image.png"
)

# ---------------------------------------------------------
# Test batch
# ---------------------------------------------------------

CASES = [
    {
        "sample_id": "sample_0001",
        "L1": 100.0,
        "L2": 40.0,
        "L3": 70.0,
        "L4": 60.0,
        "theta2_deg": 60.0
    },
    {
        "sample_id": "sample_0002",
        "L1": 100.0,
        "L2": 40.0,
        "L3": 70.0,
        "L4": 60.0,
        "theta2_deg": 131.48833
    },
    {
        "sample_id": "sample_0003",
        "L1": 100.0,
        "L2": 40.0,
        "L3": 70.0,
        "L4": 60.0,
        "theta2_deg": 131.4908168557
    },
    {
        "sample_id": "sample_0004",
        "L1": 100.0,
        "L2": 40.0,
        "L3": 70.0,
        "L4": 60.0,
        "theta2_deg": 150.0
    }
]

# ---------------------------------------------------------
# Benchmark state threshold
# ---------------------------------------------------------

NEAR_SINGULAR_RHO = 0.05

# ---------------------------------------------------------
# Full RRRR benchmark sampling settings
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

L1_FIXED = 100.0

LINK_RATIO_MIN = 0.30
LINK_RATIO_MAX = 1.20

THETA2_MIN = 0.0
THETA2_MAX = 360.0

# Reproducible random-number generator
rng = random.Random(RANDOM_SEED)

def generate_random_rrrr_candidate():

    L1 = L1_FIXED

    L2 = L1 * rng.uniform(
        LINK_RATIO_MIN,
        LINK_RATIO_MAX
    )

    L3 = L1 * rng.uniform(
        LINK_RATIO_MIN,
        LINK_RATIO_MAX
    )

    L4 = L1 * rng.uniform(
        LINK_RATIO_MIN,
        LINK_RATIO_MAX
    )

    theta2_deg = rng.uniform(
        THETA2_MIN,
        THETA2_MAX
    )

    return {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
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

        candidate = generate_random_rrrr_candidate()

        result = solve_fourbar(
            **candidate
        )

        benchmark_state = classify_rrrr_state(result)

        attempts += 1

        if benchmark_state == target_state:
            collected.append(candidate)

    return collected, attempts

def generate_exact_singular_candidate():

    while True:

        L1 = L1_FIXED

        L2 = L1 * rng.uniform(
            LINK_RATIO_MIN,
            LINK_RATIO_MAX
        )

        L3 = L1 * rng.uniform(
            LINK_RATIO_MIN,
            LINK_RATIO_MAX
        )

        L4 = L1 * rng.uniform(
            LINK_RATIO_MIN,
            LINK_RATIO_MAX
        )

        # Outer tangent condition:
        # distance BD = L3 + L4
        target_d = L3 + L4

        # Distance BD can only vary between these limits
        bd_min = abs(L1 - L2)
        bd_max = L1 + L2

        # Reject lengths if this tangent geometry is impossible
        if not (bd_min <= target_d <= bd_max):
            continue

        cos_theta2 = (
            L1**2
            + L2**2
            - target_d**2
        ) / (
            2.0 * L1 * L2
        )

        # Protect against tiny floating-point errors
        cos_theta2 = max(
            -1.0,
            min(1.0, cos_theta2)
        )

        theta2_deg = math.degrees(
            math.acos(cos_theta2)
        )

        # Randomly use the mirrored input orientation too
        if rng.random() < 0.5:
            theta2_deg = 360.0 - theta2_deg

        candidate = {
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "L4": L4,
            "theta2_deg": theta2_deg
        }

        result = solve_fourbar(
            **candidate
        )

        # Keep it only if the exact solver confirms singularity
        if classify_rrrr_state(result) == "singular":
            return candidate

def generate_near_singular_candidate():

    while True:

        # Start from a verified exact singular geometry
        singular_candidate = generate_exact_singular_candidate()

        # Try small angular perturbations around the singular input angle
        for _ in range(100):

            perturbation_deg = rng.uniform(
                0.001,
                0.05
            )

            if rng.random() < 0.5:
                perturbation_deg = -perturbation_deg

            candidate = dict(singular_candidate)

            candidate["theta2_deg"] = (
                singular_candidate["theta2_deg"]
                + perturbation_deg
            ) % 360.0

            result = solve_fourbar(
                **candidate
            )

            if classify_rrrr_state(result) == "near_singular":
                return candidate


def build_balanced_rrrr_case_pool():

    # -----------------------------------------------------
    # Regular cases
    # -----------------------------------------------------

    regular_cases, regular_attempts = (
        collect_random_cases_for_state(
            "regular",
            N_PER_STATE
        )
    )

    # -----------------------------------------------------
    # Infeasible cases
    # -----------------------------------------------------

    infeasible_cases, infeasible_attempts = (
        collect_random_cases_for_state(
            "infeasible",
            N_PER_STATE
        )
    )

    # -----------------------------------------------------
    # Exact singular cases
    # -----------------------------------------------------

    singular_cases = [
        generate_exact_singular_candidate()
        for _ in range(N_PER_STATE)
    ]

    # -----------------------------------------------------
    # Near-singular cases
    # -----------------------------------------------------

    near_singular_cases = [
        generate_near_singular_candidate()
        for _ in range(N_PER_STATE)
    ]

    # Store the balanced pool
    accepted_cases["regular"] = regular_cases
    accepted_cases["near_singular"] = near_singular_cases
    accepted_cases["singular"] = singular_cases
    accepted_cases["infeasible"] = infeasible_cases

    statistics = {
        "regular_attempts": regular_attempts,
        "infeasible_attempts": infeasible_attempts
    }

    return accepted_cases, statistics

def audit_rrrr_case_pool(case_pool):

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

            result = solve_fourbar(
                **candidate
            )

            actual_state = classify_rrrr_state(result)

            # ---------------------------------------------
            # Check state classification
            # ---------------------------------------------

            if actual_state != intended_state:
                state_errors.append(
                    (
                        intended_state,
                        index,
                        actual_state
                    )
                )

            # ---------------------------------------------
            # Check expected branch count
            # ---------------------------------------------

            if result["n_branches"] != expected_branches[intended_state]:
                branch_errors.append(
                    (
                        intended_state,
                        index,
                        result["n_branches"]
                    )
                )

            # ---------------------------------------------
            # Check geometric validation of every branch
            # ---------------------------------------------

            for solution in result["solutions"]:

                if not solution["validation_passed"]:
                    validation_errors.append(
                        (
                            intended_state,
                            index
                        )
                    )

            # ---------------------------------------------
            # Check for duplicate parameter sets
            # ---------------------------------------------

            parameter_key = (
                round(candidate["L1"], 10),
                round(candidate["L2"], 10),
                round(candidate["L3"], 10),
                round(candidate["L4"], 10),
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

    audit_report = {
        "total_cases": total_cases,
        "state_errors": state_errors,
        "branch_errors": branch_errors,
        "validation_errors": validation_errors,
        "duplicate_cases": duplicate_cases
    }

    return audit_report

def export_rrrr_case_pool(case_pool):

    os.makedirs(FINAL_RRRR_ROOT, exist_ok=True)

    # -----------------------------------------------------
    # Flatten the four state groups
    # -----------------------------------------------------

    export_cases = []

    for benchmark_state, cases in case_pool.items():

        for candidate in cases:

            export_cases.append(
                {
                    "benchmark_state": benchmark_state,
                    "candidate": candidate
                }
            )

    # -----------------------------------------------------
    # Shuffle reproducibly so sample ID does not reveal state
    # -----------------------------------------------------

    export_rng = random.Random(EXPORT_SHUFFLE_SEED)

    export_rng.shuffle(export_cases)

    # -----------------------------------------------------
    # Export every sample
    # -----------------------------------------------------

    for sample_number, item in enumerate(
        export_cases,
        start=1
    ):

        sample_id = f"sample_{sample_number:04d}"

        benchmark_state = item["benchmark_state"]
        candidate = item["candidate"]

        L1 = candidate["L1"]
        L2 = candidate["L2"]
        L3 = candidate["L3"]
        L4 = candidate["L4"]
        theta2_deg = candidate["theta2_deg"]

        # -------------------------------------------------
        # Solve again during export
        # -------------------------------------------------

        result = solve_fourbar(
            **candidate
        )

        verified_state = classify_rrrr_state(result)

        if verified_state != benchmark_state:
            raise RuntimeError(
                f"{sample_id}: expected state "
                f"'{benchmark_state}' but solver returned "
                f"'{verified_state}'."
            )

        # -------------------------------------------------
        # Create sample folder
        # -------------------------------------------------

        sample_dir = os.path.join(
            FINAL_RRRR_ROOT,
            sample_id
        )

        os.makedirs(
            sample_dir,
            exist_ok=True
        )

        # -------------------------------------------------
        # Ground-truth JSON
        # -------------------------------------------------

        ground_truth = dict(result)

        ground_truth["sample_id"] = sample_id
        ground_truth["mechanism_family"] = "RRRR"

        ground_truth["input_parameters"] = {
            "L1_mm": L1,
            "L2_mm": L2,
            "L3_mm": L3,
            "L4_mm": L4,
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

        with open(
            ground_truth_path,
            "w"
        ) as f:

            json.dump(
                ground_truth,
                f,
                indent=4
            )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt_text = f"""The image shows a planar RRRR four-bar mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L1 = {L1:g} mm
L2 = {L2:g} mm
L3 = {L3:g} mm
L4 = {L4:g} mm
theta2 = {theta2_deg:g} degrees

Angle convention:
theta2 is measured counter-clockwise from the positive x-axis.
theta3 is the orientation of link BC measured from the positive x-axis.
theta4 is the orientation of link DC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. theta3 and theta4 for every valid configuration.

If no physically valid configuration exists, report zero configurations.
"""

        prompt_path = os.path.join(
            sample_dir,
            "prompt.txt"
        )

        with open(
            prompt_path,
            "w"
        ) as f:

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
        "Final RRRR samples exported:",
        len(export_cases)
    )


def audit_exported_rrrr_dataset():

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
        for name in os.listdir(FINAL_RRRR_ROOT)
        if name.startswith("sample_")
        and os.path.isdir(
            os.path.join(FINAL_RRRR_ROOT, name)
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
            FINAL_RRRR_ROOT,
            sample_folder
        )

        # -------------------------------------------------
        # Check required files
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Read exported ground truth
        # -------------------------------------------------

        ground_truth_path = os.path.join(
            sample_dir,
            "ground_truth.json"
        )

        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        # -------------------------------------------------
        # Check sample identity
        # -------------------------------------------------

        if ground_truth["sample_id"] != sample_folder:
            sample_id_errors.append(
                sample_folder
            )

        if ground_truth["mechanism_family"] != "RRRR":
            mechanism_errors.append(
                sample_folder
            )

        # -------------------------------------------------
        # Recover input parameters
        # -------------------------------------------------

        inputs = ground_truth["input_parameters"]

        candidate = {
            "L1": inputs["L1_mm"],
            "L2": inputs["L2_mm"],
            "L3": inputs["L3_mm"],
            "L4": inputs["L4_mm"],
            "theta2_deg": inputs["theta2_deg"]
        }

        intended_state = ground_truth["benchmark_state"]

        # -------------------------------------------------
        # Re-solve from exported parameters
        # -------------------------------------------------

        result = solve_fourbar(
            **candidate
        )

        actual_state = classify_rrrr_state(
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

        # -------------------------------------------------
        # Check branch count
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Check every returned configuration
        # -------------------------------------------------

        for solution in result["solutions"]:

            if not solution["validation_passed"]:
                validation_errors.append(
                    sample_folder
                )

        # -------------------------------------------------
        # Duplicate parameter check
        # -------------------------------------------------

        parameter_key = (
            round(candidate["L1"], 10),
            round(candidate["L2"], 10),
            round(candidate["L3"], 10),
            round(candidate["L4"], 10),
            round(candidate["theta2_deg"], 10)
        )

        if parameter_key in seen_parameters:
            duplicate_cases.append(
                sample_folder
            )

        seen_parameters.add(parameter_key)

    # -----------------------------------------------------
    # Final disk audit report
    # -----------------------------------------------------

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

def classify_rrrr_state(result):

    # Excluded pathological geometry
    if result["status"] == "degenerate_excluded":
        return "excluded"

    # No physically valid assembly configuration
    if result["feasible"] is False:
        return "infeasible"

    # Exact limiting / tangent configuration
    if result["status"] == "tangent":
        return "singular"

    # Regular feasible case with two branches
    if result["n_branches"] == 2:

        rho_values = [
            solution["singularity_score"]
            for solution in result["solutions"]
        ]

        min_rho = min(rho_values)

        if min_rho <= NEAR_SINGULAR_RHO:
            return "near_singular"

        return "regular"

    # Safety fallback for any unexpected solver state
    return "unclassified"
print("Number of test cases:", len(CASES))
print("Output folder:", OUTPUT_ROOT)
print("Canonical image:", CANONICAL_IMAGE)

# ---------------------------------------------------------
# Generate all test samples automatically
# ---------------------------------------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for case in CASES:

    sample_id = case["sample_id"]

    L1 = case["L1"]
    L2 = case["L2"]
    L3 = case["L3"]
    L4 = case["L4"]
    theta2_deg = case["theta2_deg"]

    # Create sample folder
    sample_dir = os.path.join(
        OUTPUT_ROOT,
        sample_id
    )

    os.makedirs(sample_dir, exist_ok=True)

    # -----------------------------------------------------
    # Exact solver ground truth
    # -----------------------------------------------------

    result = solve_fourbar(
        L1=L1,
        L2=L2,
        L3=L3,
        L4=L4,
        theta2_deg=theta2_deg
    )
    benchmark_state = classify_rrrr_state(result)
    ground_truth = dict(result)

    ground_truth["benchmark_state"] = benchmark_state
    ground_truth["near_singular_rho_threshold"] = NEAR_SINGULAR_RHO
    ground_truth_path = os.path.join(
        sample_dir,
        "ground_truth.json"
    )

    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth, f, indent=4)

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt_text = f"""The image shows a planar RRRR four-bar mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L1 = {L1:g} mm
L2 = {L2:g} mm
L3 = {L3:g} mm
L4 = {L4:g} mm
theta2 = {theta2_deg:g} degrees

Angle convention:
theta2 is measured counter-clockwise from the positive x-axis.
theta3 is the orientation of link BC measured from the positive x-axis.
theta4 is the orientation of link DC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. theta3 and theta4 for every valid configuration.

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

    # -----------------------------------------------------
    # Progress report
    # -----------------------------------------------------

    print(
    sample_id,
    "| solver status =", result["status"],
    "| benchmark state =", benchmark_state,
    "| configurations =", result["n_branches"]
)


print("Batch generation complete.")

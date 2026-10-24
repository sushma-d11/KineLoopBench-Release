import os
import json
import shutil
import random
import math

from planar_5r_solver import solve_planar_5r


SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)


OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "5R_Auto_Test"
)

FINAL_5R_ROOT = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "KineBench_5R"
)

CANONICAL_IMAGE = os.path.join(
    PROJECT_ROOT,
    "04_Dataset",
    "Pilot_5R",
    "sample_0001",
    "image.png"
)


print(
    "Output folder:",
    OUTPUT_ROOT
)

print(
    "Canonical image:",
    CANONICAL_IMAGE
)

NEAR_SINGULAR_RHO = 0.05

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

THETA_MIN = 0.0
THETA_MAX = 360.0

rng = random.Random(
    RANDOM_SEED
)


def classify_5r_state(result):

    if result["status"] == "degenerate_excluded":
        return "excluded"

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

CASES = [

    {
        "sample_id": "sample_0001",
        "L1": 100,
        "L2": 40,
        "L3": 70,
        "L4": 70,
        "L5": 40,
        "theta2_deg": 60,
        "theta5_deg": 120
    },

    {
        "sample_id": "sample_0002",
        "L1": 100,
        "L2": 40,
        "L3": 10.001,
        "L4": 10.001,
        "L5": 40,
        "theta2_deg": 0,
        "theta5_deg": 180
    },

    {
        "sample_id": "sample_0003",
        "L1": 100,
        "L2": 40,
        "L3": 10,
        "L4": 10,
        "L5": 40,
        "theta2_deg": 0,
        "theta5_deg": 180
    },

    {
        "sample_id": "sample_0004",
        "L1": 100,
        "L2": 40,
        "L3": 10,
        "L4": 10,
        "L5": 40,
        "theta2_deg": 60,
        "theta5_deg": 120
    }
]


def generate_random_5r_candidate():

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

    L5 = L1 * rng.uniform(
        LINK_RATIO_MIN,
        LINK_RATIO_MAX
    )

    theta2_deg = rng.uniform(
        THETA_MIN,
        THETA_MAX
    )

    theta5_deg = rng.uniform(
        THETA_MIN,
        THETA_MAX
    )

    return {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5,
        "theta2_deg": theta2_deg,
        "theta5_deg": theta5_deg
    }

def collect_random_cases_for_state(
    target_state,
    n_needed
):

    collected = []
    attempts = 0

    max_attempts = 100000

    while len(collected) < n_needed:

        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not collect enough "
                f"'{target_state}' cases."
            )

        candidate = (
            generate_random_5r_candidate()
        )

        result = solve_planar_5r(
            **candidate
        )

        benchmark_state = (
            classify_5r_state(result)
        )

        attempts += 1

        if benchmark_state == target_state:
            collected.append(candidate)

    return collected, attempts

def generate_exact_singular_5r_candidate():

    while True:

        L1 = L1_FIXED

        L2 = L1 * rng.uniform(
            LINK_RATIO_MIN,
            LINK_RATIO_MAX
        )

        L5 = L1 * rng.uniform(
            LINK_RATIO_MIN,
            LINK_RATIO_MAX
        )

        theta2_deg = rng.uniform(
            THETA_MIN,
            THETA_MAX
        )

        theta5_deg = rng.uniform(
            THETA_MIN,
            THETA_MAX
        )

        theta2 = math.radians(
            theta2_deg
        )

        theta5 = math.radians(
            theta5_deg
        )

        # Moving crank endpoints
        Bx = L2 * math.cos(theta2)
        By = L2 * math.sin(theta2)

        Ex = (
            L1
            + L5 * math.cos(theta5)
        )

        Ey = (
            L5 * math.sin(theta5)
        )

        distance_BE = math.hypot(
            Ex - Bx,
            Ey - By
        )

        # Allowed moving-link range
        link_min = (
            L1 * LINK_RATIO_MIN
        )

        link_max = (
            L1 * LINK_RATIO_MAX
        )

        # For outer tangency:
        # L3 + L4 = distance_BE
        L3_min = max(
            link_min,
            distance_BE - link_max
        )

        L3_max = min(
            link_max,
            distance_BE - link_min
        )

        # No valid split inside benchmark range
        if L3_min > L3_max:
            continue

        L3 = rng.uniform(
            L3_min,
            L3_max
        )

        L4 = (
            distance_BE - L3
        )

        candidate = {
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "L4": L4,
            "L5": L5,
            "theta2_deg": theta2_deg,
            "theta5_deg": theta5_deg
        }

        result = solve_planar_5r(
            **candidate
        )

        # Keep only solver-confirmed exact singular cases
        if classify_5r_state(result) == "singular":
            return candidate

def generate_near_singular_5r_candidate():

    while True:

        # Start from an exact singular case
        base_candidate = (
            generate_exact_singular_5r_candidate()
        )

        # Very small controlled perturbation
        perturb_fraction = rng.uniform(
            0.00001,
            0.00015
        )

        candidate = dict(
            base_candidate
        )

        candidate["L3"] = (
            base_candidate["L3"]
            * (1.0 + perturb_fraction)
        )

        candidate["L4"] = (
            base_candidate["L4"]
            * (1.0 + perturb_fraction)
        )

        # Keep link lengths inside benchmark range
        link_max = (
            L1_FIXED * LINK_RATIO_MAX
        )

        if (
            candidate["L3"] > link_max
            or candidate["L4"] > link_max
        ):
            continue

        result = solve_planar_5r(
            **candidate
        )

        # Keep only solver-confirmed near-singular cases
        if classify_5r_state(result) == "near_singular":
            return candidate


def build_balanced_5r_case_pool():

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
        generate_exact_singular_5r_candidate()
        for _ in range(N_PER_STATE)
    ]

    # Near-singular cases
    near_singular_cases = [
        generate_near_singular_5r_candidate()
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

def audit_5r_case_pool(case_pool):

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

        for index, candidate in enumerate(
            cases,
            start=1
        ):

            total_cases += 1

            result = solve_planar_5r(
                **candidate
            )

            actual_state = classify_5r_state(
                result
            )

            # Check benchmark state
            if actual_state != intended_state:

                state_errors.append(
                    (
                        intended_state,
                        index,
                        actual_state
                    )
                )

            # Check number of configurations
            if (
                result["n_branches"]
                != expected_branches[intended_state]
            ):

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
                round(candidate["L1"], 10),
                round(candidate["L2"], 10),
                round(candidate["L3"], 10),
                round(candidate["L4"], 10),
                round(candidate["L5"], 10),
                round(candidate["theta2_deg"], 10),
                round(candidate["theta5_deg"], 10)
            )

            if parameter_key in seen_parameters:

                duplicate_cases.append(
                    (
                        intended_state,
                        index
                    )
                )

            seen_parameters.add(
                parameter_key
            )

    return {
        "total_cases": total_cases,
        "state_errors": state_errors,
        "branch_errors": branch_errors,
        "validation_errors": validation_errors,
        "duplicate_cases": duplicate_cases
    }


def export_5r_case_pool(case_pool):

    os.makedirs(
        FINAL_5R_ROOT,
        exist_ok=True
    )

    export_cases = []

    # Flatten all state groups
    for benchmark_state, cases in case_pool.items():

        for candidate in cases:

            export_cases.append(
                {
                    "benchmark_state": benchmark_state,
                    "candidate": candidate
                }
            )

    # Shuffle so sample ID does not reveal state
    export_rng = random.Random(
        EXPORT_SHUFFLE_SEED
    )

    export_rng.shuffle(
        export_cases
    )

    for sample_number, item in enumerate(
        export_cases,
        start=1
    ):

        sample_id = (
            f"sample_{sample_number:04d}"
        )

        benchmark_state = (
            item["benchmark_state"]
        )

        candidate = (
            item["candidate"]
        )

        # Re-solve during export
        result = solve_planar_5r(
            **candidate
        )

        verified_state = classify_5r_state(
            result
        )

        if verified_state != benchmark_state:

            raise RuntimeError(
                f"{sample_id}: expected "
                f"'{benchmark_state}' but solver "
                f"returned '{verified_state}'."
            )

        sample_dir = os.path.join(
            FINAL_5R_ROOT,
            sample_id
        )

        os.makedirs(
            sample_dir,
            exist_ok=True
        )

        # ---------------------------------------------
        # Ground truth
        # ---------------------------------------------

        ground_truth = dict(
            result
        )

        ground_truth["sample_id"] = (
            sample_id
        )

        ground_truth["mechanism_family"] = (
            "5R"
        )

        ground_truth["input_parameters"] = {
            "L1_mm": candidate["L1"],
            "L2_mm": candidate["L2"],
            "L3_mm": candidate["L3"],
            "L4_mm": candidate["L4"],
            "L5_mm": candidate["L5"],
            "theta2_deg": candidate["theta2_deg"],
            "theta5_deg": candidate["theta5_deg"]
        }

        ground_truth["benchmark_state"] = (
            benchmark_state
        )

        ground_truth[
            "near_singular_rho_threshold"
        ] = NEAR_SINGULAR_RHO

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

        # ---------------------------------------------
        # Prompt
        # ---------------------------------------------

        prompt_text = f"""The image shows a planar 5R five-bar mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L1 = {candidate["L1"]:g} mm
L2 = {candidate["L2"]:g} mm
L3 = {candidate["L3"]:g} mm
L4 = {candidate["L4"]:g} mm
L5 = {candidate["L5"]:g} mm
theta2 = {candidate["theta2_deg"]:g} degrees
theta5 = {candidate["theta5_deg"]:g} degrees

Angle convention:
theta2 and theta5 are measured counter-clockwise from the positive x-axis.
theta3 is the orientation of link BC measured from the positive x-axis.
theta4 is the orientation of link EC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. C = (xC, yC), theta3, and theta4 for every valid configuration.

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

            f.write(
                prompt_text
            )

        # ---------------------------------------------
        # Canonical image
        # ---------------------------------------------

        image_output_path = os.path.join(
            sample_dir,
            "image.png"
        )

        shutil.copy2(
            CANONICAL_IMAGE,
            image_output_path
        )

    print(
        "Final 5R samples exported:",
        len(export_cases)
    )

def audit_exported_5r_dataset():

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

    expected_branches = {
        "regular": 2,
        "near_singular": 2,
        "singular": 1,
        "infeasible": 0
    }

    sample_folders = sorted(
        name
        for name in os.listdir(FINAL_5R_ROOT)
        if name.startswith("sample_")
        and os.path.isdir(
            os.path.join(
                FINAL_5R_ROOT,
                name
            )
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
            FINAL_5R_ROOT,
            sample_folder
        )

        existing_files = set(
            os.listdir(sample_dir)
        )

        missing = (
            expected_files
            - existing_files
        )

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

        with open(
            ground_truth_path,
            "r"
        ) as f:

            ground_truth = json.load(f)

        # Check sample ID
        if (
            ground_truth["sample_id"]
            != sample_folder
        ):

            sample_id_errors.append(
                sample_folder
            )

        # Check mechanism family
        if (
            ground_truth["mechanism_family"]
            != "5R"
        ):

            mechanism_errors.append(
                sample_folder
            )

        inputs = ground_truth[
            "input_parameters"
        ]

        candidate = {
            "L1": inputs["L1_mm"],
            "L2": inputs["L2_mm"],
            "L3": inputs["L3_mm"],
            "L4": inputs["L4_mm"],
            "L5": inputs["L5_mm"],
            "theta2_deg": inputs["theta2_deg"],
            "theta5_deg": inputs["theta5_deg"]
        }

        intended_state = ground_truth[
            "benchmark_state"
        ]

        # Independently re-solve
        result = solve_planar_5r(
            **candidate
        )

        actual_state = classify_5r_state(
            result
        )

        # Check state
        if actual_state != intended_state:

            state_errors.append(
                (
                    sample_folder,
                    intended_state,
                    actual_state
                )
            )

        if intended_state in state_counts:

            state_counts[
                intended_state
            ] += 1

        # Check branch count
        if (
            result["n_branches"]
            != expected_branches[
                intended_state
            ]
        ):

            branch_errors.append(
                sample_folder
            )

        # Check every returned configuration
        for solution in result["solutions"]:

            if not solution[
                "validation_passed"
            ]:

                validation_errors.append(
                    sample_folder
                )

        # Check duplicate parameter sets
        parameter_key = (
            round(candidate["L1"], 10),
            round(candidate["L2"], 10),
            round(candidate["L3"], 10),
            round(candidate["L4"], 10),
            round(candidate["L5"], 10),
            round(candidate["theta2_deg"], 10),
            round(candidate["theta5_deg"], 10)
        )

        if parameter_key in seen_parameters:

            duplicate_cases.append(
                sample_folder
            )

        seen_parameters.add(
            parameter_key
        )

    report = {
        "folder_count": len(
            sample_folders
        ),

        "state_counts": state_counts,

        "expected_state_counts":
            expected_state_counts,

        "missing_files":
            missing_files,

        "sample_id_errors":
            sample_id_errors,

        "mechanism_errors":
            mechanism_errors,

        "state_errors":
            state_errors,

        "branch_errors":
            branch_errors,

        "validation_errors":
            validation_errors,

        "duplicate_cases":
            duplicate_cases
    }

    return report

for case in CASES:

    sample_id = case["sample_id"]

    parameters = {
        "L1": case["L1"],
        "L2": case["L2"],
        "L3": case["L3"],
        "L4": case["L4"],
        "L5": case["L5"],
        "theta2_deg": case["theta2_deg"],
        "theta5_deg": case["theta5_deg"]
    }

    result = solve_planar_5r(
        **parameters
    )

    benchmark_state = classify_5r_state(
        result
    )

    print(
        sample_id,
        "| solver status =",
        result["status"],
        "| benchmark state =",
        benchmark_state,
        "| configurations =",
        result["n_branches"]
    )

# -------------------------------------------------
# Automatic packaging test
# -------------------------------------------------

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


for case in CASES:

    sample_id = case["sample_id"]

    parameters = {
        "L1": case["L1"],
        "L2": case["L2"],
        "L3": case["L3"],
        "L4": case["L4"],
        "L5": case["L5"],
        "theta2_deg": case["theta2_deg"],
        "theta5_deg": case["theta5_deg"]
    }

    result = solve_planar_5r(
        **parameters
    )

    benchmark_state = classify_5r_state(
        result
    )

    sample_dir = os.path.join(
        OUTPUT_ROOT,
        sample_id
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    # ---------------------------------------------
    # Ground truth
    # ---------------------------------------------

    ground_truth = dict(result)

    ground_truth["sample_id"] = sample_id
    ground_truth["mechanism_family"] = "5R"

    ground_truth["input_parameters"] = {
        "L1_mm": case["L1"],
        "L2_mm": case["L2"],
        "L3_mm": case["L3"],
        "L4_mm": case["L4"],
        "L5_mm": case["L5"],
        "theta2_deg": case["theta2_deg"],
        "theta5_deg": case["theta5_deg"]
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

    # ---------------------------------------------
    # Prompt
    # ---------------------------------------------

    prompt_text = f"""The image shows a planar 5R five-bar mechanism.
The diagram is schematic and not drawn to scale; use the numerical parameters below for all calculations.

Numerical parameters:
L1 = {case["L1"]:g} mm
L2 = {case["L2"]:g} mm
L3 = {case["L3"]:g} mm
L4 = {case["L4"]:g} mm
L5 = {case["L5"]:g} mm
theta2 = {case["theta2_deg"]:g} degrees
theta5 = {case["theta5_deg"]:g} degrees

Angle convention:
theta2 and theta5 are measured counter-clockwise from the positive x-axis.
theta3 is the orientation of link BC measured from the positive x-axis.
theta4 is the orientation of link EC measured from the positive x-axis.
Report angles in the range 0 to 360 degrees.

Task:
Determine the complete set of physically valid assembly configurations for this input state.

Report:
1. The total number of valid configurations.
2. C = (xC, yC), theta3, and theta4 for every valid configuration.

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

    # ---------------------------------------------
    # Canonical image
    # ---------------------------------------------

    shutil.copy2(
        CANONICAL_IMAGE,
        os.path.join(
            sample_dir,
            "image.png"
        )
    )

    print(
        sample_id,
        "| solver status =",
        result["status"],
        "| benchmark state =",
        benchmark_state,
        "| configurations =",
        result["n_branches"]
    )


print(
    "5R automatic test packaging complete."
)

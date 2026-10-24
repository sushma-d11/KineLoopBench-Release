import json
import math
import itertools


# =================================================
# Frozen primary scoring thresholds
# =================================================

POSITION_TOL_MM = 1.0

ANGLE_TOL_DEG = 1.0

PHYSICAL_RESIDUAL_TOL = 0.01


# =================================================
# 1. Response parsing
# =================================================

def try_strict_json(raw_text):

    text = raw_text.strip()

    try:

        parsed = json.loads(text)

        if not isinstance(
            parsed,
            dict
        ):
            return False, None

        return True, parsed

    except (
        json.JSONDecodeError,
        TypeError
    ):

        return False, None


def remove_simple_code_fence(
    raw_text
):

    text = raw_text.strip()

    if not text.startswith("```"):
        return None

    if not text.endswith("```"):
        return None

    lines = text.splitlines()

    if len(lines) < 3:
        return None

    first_line = (
        lines[0]
        .strip()
        .lower()
    )

    if first_line not in [
        "```",
        "```json"
    ]:
        return None

    if lines[-1].strip() != "```":
        return None

    return "\n".join(
        lines[1:-1]
    ).strip()


def parse_model_response(
    raw_text
):

    strict_format, parsed = (
        try_strict_json(
            raw_text
        )
    )

    if strict_format:

        return {
            "strict_format": True,
            "parseable": True,
            "fence_removed": False,
            "parsed_response": parsed
        }

    cleaned = (
        remove_simple_code_fence(
            raw_text
        )
    )

    if cleaned is None:

        return {
            "strict_format": False,
            "parseable": False,
            "fence_removed": False,
            "parsed_response": None
        }

    cleaned_ok, cleaned_parsed = (
        try_strict_json(
            cleaned
        )
    )

    if cleaned_ok:

        return {
            "strict_format": False,
            "parseable": True,
            "fence_removed": True,
            "parsed_response":
                cleaned_parsed
        }

    return {
        "strict_format": False,
        "parseable": False,
        "fence_removed": True,
        "parsed_response": None
    }


# =================================================
# 2. Schema checks
# =================================================

def check_basic_schema(
    parsed_response
):

    if not isinstance(
        parsed_response,
        dict
    ):
        return False

    if (
        "configuration_count"
        not in parsed_response
    ):
        return False

    if (
        "configurations"
        not in parsed_response
    ):
        return False

    count = parsed_response[
        "configuration_count"
    ]

    configurations = parsed_response[
        "configurations"
    ]

    if (
        not isinstance(count, int)
        or isinstance(count, bool)
    ):
        return False

    if count < 0:
        return False

    if not isinstance(
        configurations,
        list
    ):
        return False

    for configuration in configurations:

        if not isinstance(
            configuration,
            dict
        ):
            return False

    return True


def check_internal_consistency(
    parsed_response
):

    if not check_basic_schema(
        parsed_response
    ):
        return False

    return (
        parsed_response[
            "configuration_count"
        ]
        ==
        len(
            parsed_response[
                "configurations"
            ]
        )
    )


# =================================================
# 3. Required fields by mechanism family
# =================================================

def required_configuration_fields(
    mechanism_family
):

    if mechanism_family in [
        "RRRR",
        "5R"
    ]:

        return [
            "xC_mm",
            "yC_mm",
            "theta3_deg",
            "theta4_deg"
        ]

    if mechanism_family == "RRRP":

        return [
            "xC_mm",
            "theta3_deg"
        ]

    raise ValueError(
        "Unknown mechanism family: "
        + str(mechanism_family)
    )


def configuration_has_required_fields(
    configuration,
    mechanism_family
):

    required_fields = (
        required_configuration_fields(
            mechanism_family
        )
    )

    for field in required_fields:

        if field not in configuration:
            return False

        value = configuration[
            field
        ]

        if (
            not isinstance(
                value,
                (int, float)
            )
            or isinstance(
                value,
                bool
            )
            or not math.isfinite(
                float(value)
            )
        ):
            return False

    return True


# =================================================
# 4. Angle error
# =================================================

def circular_angle_error(
    predicted,
    truth
):

    difference = (
        abs(
            float(predicted)
            - float(truth)
        )
        % 360.0
    )

    return min(
        difference,
        360.0 - difference
    )


# =================================================
# 5. Branch numerical errors
# =================================================

def branch_errors(
    predicted,
    truth,
    mechanism_family
):

    if mechanism_family in [
        "RRRR",
        "5R"
    ]:

        dx = (
            float(
                predicted["xC_mm"]
            )
            - float(
                truth["xC_mm"]
            )
        )

        dy = (
            float(
                predicted["yC_mm"]
            )
            - float(
                truth["yC_mm"]
            )
        )

        position_error = (
            math.hypot(
                dx,
                dy
            )
        )

        theta3_error = (
            circular_angle_error(
                predicted[
                    "theta3_deg"
                ],
                truth[
                    "theta3_deg"
                ]
            )
        )

        theta4_error = (
            circular_angle_error(
                predicted[
                    "theta4_deg"
                ],
                truth[
                    "theta4_deg"
                ]
            )
        )

        return {
            "position_error_mm":
                position_error,

            "theta3_error_deg":
                theta3_error,

            "theta4_error_deg":
                theta4_error
        }


    if mechanism_family == "RRRP":

        position_error = abs(
            float(
                predicted["xC_mm"]
            )
            - float(
                truth["xC_mm"]
            )
        )

        theta3_error = (
            circular_angle_error(
                predicted[
                    "theta3_deg"
                ],
                truth[
                    "theta3_deg"
                ]
            )
        )

        return {
            "position_error_mm":
                position_error,

            "theta3_error_deg":
                theta3_error
        }


    raise ValueError(
        "Unknown mechanism family."
    )


# =================================================
# 6. Branch matching cost
# =================================================

def branch_matching_cost(
    predicted,
    truth,
    mechanism_family
):

    errors = branch_errors(
        predicted,
        truth,
        mechanism_family
    )

    cost = (
        errors[
            "position_error_mm"
        ]
        / POSITION_TOL_MM
    )

    cost += (
        errors[
            "theta3_error_deg"
        ]
        / ANGLE_TOL_DEG
    )

    if mechanism_family in [
        "RRRR",
        "5R"
    ]:

        cost += (
            errors[
                "theta4_error_deg"
            ]
            / ANGLE_TOL_DEG
        )

    return cost


# =================================================
# 7. Order-independent branch assignment
# =================================================

def find_best_matching(
    predicted_configurations,
    ground_truth_configurations,
    mechanism_family
):

    n_pred = len(
        predicted_configurations
    )

    n_gt = len(
        ground_truth_configurations
    )

    n_match = min(
        n_pred,
        n_gt
    )

    if n_match == 0:
        return []

    best_matching = None
    best_total_cost = None

    for pred_subset in itertools.combinations(
        range(n_pred),
        n_match
    ):

        for gt_subset in itertools.combinations(
            range(n_gt),
            n_match
        ):

            for gt_permutation in itertools.permutations(
                gt_subset
            ):

                current_pairs = []
                current_cost = 0.0

                for (
                    pred_index,
                    gt_index
                ) in zip(
                    pred_subset,
                    gt_permutation
                ):

                    cost = (
                        branch_matching_cost(
                            predicted_configurations[
                                pred_index
                            ],
                            ground_truth_configurations[
                                gt_index
                            ],
                            mechanism_family
                        )
                    )

                    current_cost += cost

                    current_pairs.append(
                        (
                            pred_index,
                            gt_index
                        )
                    )

                if (
                    best_total_cost
                    is None
                    or current_cost
                    < best_total_cost
                ):

                    best_total_cost = (
                        current_cost
                    )

                    best_matching = (
                        current_pairs
                    )

    return best_matching


# =================================================
# 8. Numerical branch correctness
# =================================================

def branch_is_correct(
    predicted,
    truth,
    mechanism_family
):

    errors = branch_errors(
        predicted,
        truth,
        mechanism_family
    )

    correct = (
        errors[
            "position_error_mm"
        ]
        <= POSITION_TOL_MM

        and errors[
            "theta3_error_deg"
        ]
        <= ANGLE_TOL_DEG
    )

    if mechanism_family in [
        "RRRR",
        "5R"
    ]:

        correct = (
            correct
            and errors[
                "theta4_error_deg"
            ]
            <= ANGLE_TOL_DEG
        )

    return correct, errors


# =================================================
# 9. Physical validity: RRRR
# =================================================

def physical_residual_rrrr(
    configuration,
    inputs
):

    L1 = float(
        inputs["L1_mm"]
    )

    L2 = float(
        inputs["L2_mm"]
    )

    L3 = float(
        inputs["L3_mm"]
    )

    L4 = float(
        inputs["L4_mm"]
    )

    theta2 = math.radians(
        float(
            inputs["theta2_deg"]
        )
    )

    B = (
        L2 * math.cos(theta2),
        L2 * math.sin(theta2)
    )

    D = (
        L1,
        0.0
    )

    C = (
        float(
            configuration["xC_mm"]
        ),
        float(
            configuration["yC_mm"]
        )
    )

    theta3 = math.radians(
        float(
            configuration[
                "theta3_deg"
            ]
        )
    )

    theta4 = math.radians(
        float(
            configuration[
                "theta4_deg"
            ]
        )
    )

    C_from_BC = (
        B[0]
        + L3 * math.cos(theta3),

        B[1]
        + L3 * math.sin(theta3)
    )

    C_from_DC = (
        D[0]
        + L4 * math.cos(theta4),

        D[1]
        + L4 * math.sin(theta4)
    )

    residual_BC = math.hypot(
        C[0] - C_from_BC[0],
        C[1] - C_from_BC[1]
    )

    residual_DC = math.hypot(
        C[0] - C_from_DC[0],
        C[1] - C_from_DC[1]
    )

    loop_residual = math.hypot(
        C_from_BC[0]
        - C_from_DC[0],

        C_from_BC[1]
        - C_from_DC[1]
    )

    maximum_residual = max(
        residual_BC,
        residual_DC,
        loop_residual
    )

    L_ref = max(
        abs(L1),
        1.0
    )

    normalized_residual = (
        maximum_residual
        / L_ref
    )

    return {
        "maximum_residual_mm":
            maximum_residual,

        "normalized_residual":
            normalized_residual,

        "physically_valid":
            normalized_residual
            <= PHYSICAL_RESIDUAL_TOL
    }


# =================================================
# 10. Physical validity: RRRP
# =================================================

def physical_residual_rrrp(
    configuration,
    inputs
):

    L2 = float(
        inputs["L2_mm"]
    )

    L3 = float(
        inputs["L3_mm"]
    )

    e = float(
        inputs["e_mm"]
    )

    theta2 = math.radians(
        float(
            inputs["theta2_deg"]
        )
    )

    B = (
        L2 * math.cos(theta2),
        L2 * math.sin(theta2)
    )

    xC = float(
        configuration["xC_mm"]
    )

    C = (
        xC,
        e
    )

    theta3 = math.radians(
        float(
            configuration[
                "theta3_deg"
            ]
        )
    )

    C_from_BC = (
        B[0]
        + L3 * math.cos(theta3),

        B[1]
        + L3 * math.sin(theta3)
    )

    endpoint_residual = math.hypot(
        C[0] - C_from_BC[0],
        C[1] - C_from_BC[1]
    )

    actual_BC_length = math.hypot(
        C[0] - B[0],
        C[1] - B[1]
    )

    rod_length_residual = abs(
        actual_BC_length
        - L3
    )

    slider_residual = abs(
        C_from_BC[1]
        - e
    )

    maximum_residual = max(
        endpoint_residual,
        rod_length_residual,
        slider_residual
    )

    L_ref = max(
        abs(L3),
        1.0
    )

    normalized_residual = (
        maximum_residual
        / L_ref
    )

    return {
        "maximum_residual_mm":
            maximum_residual,

        "normalized_residual":
            normalized_residual,

        "physically_valid":
            normalized_residual
            <= PHYSICAL_RESIDUAL_TOL
    }


# =================================================
# 11. Physical validity: 5R
# =================================================

def physical_residual_5r(
    configuration,
    inputs
):

    L1 = float(
        inputs["L1_mm"]
    )

    L2 = float(
        inputs["L2_mm"]
    )

    L3 = float(
        inputs["L3_mm"]
    )

    L4 = float(
        inputs["L4_mm"]
    )

    L5 = float(
        inputs["L5_mm"]
    )

    theta2 = math.radians(
        float(
            inputs["theta2_deg"]
        )
    )

    theta5 = math.radians(
        float(
            inputs["theta5_deg"]
        )
    )

    B = (
        L2 * math.cos(theta2),
        L2 * math.sin(theta2)
    )

    E = (
        L1
        + L5 * math.cos(theta5),

        L5 * math.sin(theta5)
    )

    C = (
        float(
            configuration["xC_mm"]
        ),
        float(
            configuration["yC_mm"]
        )
    )

    theta3 = math.radians(
        float(
            configuration[
                "theta3_deg"
            ]
        )
    )

    theta4 = math.radians(
        float(
            configuration[
                "theta4_deg"
            ]
        )
    )

    C_from_BC = (
        B[0]
        + L3 * math.cos(theta3),

        B[1]
        + L3 * math.sin(theta3)
    )

    C_from_EC = (
        E[0]
        + L4 * math.cos(theta4),

        E[1]
        + L4 * math.sin(theta4)
    )

    residual_BC = math.hypot(
        C[0] - C_from_BC[0],
        C[1] - C_from_BC[1]
    )

    residual_EC = math.hypot(
        C[0] - C_from_EC[0],
        C[1] - C_from_EC[1]
    )

    loop_residual = math.hypot(
        C_from_BC[0]
        - C_from_EC[0],

        C_from_BC[1]
        - C_from_EC[1]
    )

    actual_BC_length = math.hypot(
        C[0] - B[0],
        C[1] - B[1]
    )

    actual_EC_length = math.hypot(
        C[0] - E[0],
        C[1] - E[1]
    )

    length_residual_BC = abs(
        actual_BC_length
        - L3
    )

    length_residual_EC = abs(
        actual_EC_length
        - L4
    )

    maximum_residual = max(
        residual_BC,
        residual_EC,
        loop_residual,
        length_residual_BC,
        length_residual_EC
    )

    L_ref = max(
        abs(L1),
        1.0
    )

    normalized_residual = (
        maximum_residual
        / L_ref
    )

    return {
        "maximum_residual_mm":
            maximum_residual,

        "normalized_residual":
            normalized_residual,

        "physically_valid":
            normalized_residual
            <= PHYSICAL_RESIDUAL_TOL
    }


# =================================================
# 12. Family dispatcher
# =================================================

def physical_residual(
    configuration,
    inputs,
    mechanism_family
):

    if mechanism_family == "RRRR":

        return physical_residual_rrrr(
            configuration,
            inputs
        )

    if mechanism_family == "RRRP":

        return physical_residual_rrrp(
            configuration,
            inputs
        )

    if mechanism_family == "5R":

        return physical_residual_5r(
            configuration,
            inputs
        )

    raise ValueError(
        "Unknown mechanism family: "
        + str(mechanism_family)
    )
# =================================================
# 13. Convert solver ground truth to scoring format
# =================================================

def extract_ground_truth_configurations(
    ground_truth,
    mechanism_family
):

    solutions = ground_truth.get(
        "solutions",
        []
    )

    configurations = []


    for solution in solutions:

        C = solution[
            "C"
        ]


        if mechanism_family in [
            "RRRR",
            "5R"
        ]:

            configuration = {

                "xC_mm":
                    float(C[0]),

                "yC_mm":
                    float(C[1]),

                "theta3_deg":
                    float(
                        solution[
                            "theta3_deg"
                        ]
                    ),

                "theta4_deg":
                    float(
                        solution[
                            "theta4_deg"
                        ]
                    )
            }


        elif mechanism_family == "RRRP":

            configuration = {

                "xC_mm":
                    float(C[0]),

                "theta3_deg":
                    float(
                        solution[
                            "theta3_deg"
                        ]
                    )
            }


        else:

            raise ValueError(
                "Unknown mechanism family: "
                + str(mechanism_family)
            )


        configurations.append(
            configuration
        )


    return configurations

# =================================================
# 14. Unified response scorer
# =================================================

def score_response(
    raw_text,
    ground_truth,
    mechanism_family
):

    # ---------------------------------------------
    # Ground truth
    # ---------------------------------------------

    ground_truth_configurations = (
        extract_ground_truth_configurations(
            ground_truth,
            mechanism_family
        )
    )

    ground_truth_count = len(
        ground_truth_configurations
    )


    result = {

        "mechanism_family":
            mechanism_family,

        "benchmark_state":
            ground_truth.get(
                "benchmark_state"
            ),

        "ground_truth_count":
            ground_truth_count,

        "strict_format":
            False,

        "parseable":
            False,

        "fence_removed":
            False,

        "schema_valid":
            False,

        "internal_consistency":
            False,

        "configuration_fields_valid":
            False,

        "reported_count":
            None,

        "listed_count":
            None,

        "configuration_count_correct":
            False,

        "matched_branch_count":
            0,

        "correct_matched_branches":
            0,

        "all_gt_branches_correct":
            False,

        "extra_predicted_branches":
            None,

        "no_extra_predicted_branches":
            False,

        "complete_set_correct":
            False,

        "all_reported_physically_valid":
            False,

        "physical_exact_success":
            False,

        "strict_task_success":
            False,

        "matched_branch_details":
            [],

        "physical_validity_details":
            []
    }


    # ---------------------------------------------
    # Parse
    # ---------------------------------------------

    parse_result = parse_model_response(
        raw_text
    )


    result["strict_format"] = (
        parse_result[
            "strict_format"
        ]
    )

    result["parseable"] = (
        parse_result[
            "parseable"
        ]
    )

    result["fence_removed"] = (
        parse_result[
            "fence_removed"
        ]
    )


    if not result["parseable"]:

        return result


    parsed_response = (
        parse_result[
            "parsed_response"
        ]
    )


    # ---------------------------------------------
    # Basic schema
    # ---------------------------------------------

    schema_valid = check_basic_schema(
        parsed_response
    )


    result["schema_valid"] = (
        schema_valid
    )


    if not schema_valid:

        return result


    # ---------------------------------------------
    # Count information
    # ---------------------------------------------

    reported_count = (
        parsed_response[
            "configuration_count"
        ]
    )


    predicted_configurations = (
        parsed_response[
            "configurations"
        ]
    )


    listed_count = len(
        predicted_configurations
    )


    result["reported_count"] = (
        reported_count
    )

    result["listed_count"] = (
        listed_count
    )


    result[
        "internal_consistency"
    ] = check_internal_consistency(
        parsed_response
    )


    result[
        "configuration_count_correct"
    ] = (
        reported_count
        == ground_truth_count
    )


    # ---------------------------------------------
    # Required numerical fields
    # ---------------------------------------------

    fields_valid = all(

        configuration_has_required_fields(
            configuration,
            mechanism_family
        )

        for configuration
        in predicted_configurations
    )


    result[
        "configuration_fields_valid"
    ] = fields_valid


    if not fields_valid:

        return result


    # ---------------------------------------------
    # Order-independent matching
    # ---------------------------------------------

    matching = find_best_matching(
        predicted_configurations,
        ground_truth_configurations,
        mechanism_family
    )


    result[
        "matched_branch_count"
    ] = len(
        matching
    )


    correct_matched_branches = 0

    correct_gt_indices = set()


    for (
        predicted_index,
        ground_truth_index
    ) in matching:

        predicted_configuration = (
            predicted_configurations[
                predicted_index
            ]
        )


        truth_configuration = (
            ground_truth_configurations[
                ground_truth_index
            ]
        )


        branch_correct, errors = (
            branch_is_correct(
                predicted_configuration,
                truth_configuration,
                mechanism_family
            )
        )


        if branch_correct:

            correct_matched_branches += 1

            correct_gt_indices.add(
                ground_truth_index
            )


        result[
            "matched_branch_details"
        ].append(
            {
                "predicted_index":
                    predicted_index,

                "ground_truth_index":
                    ground_truth_index,

                "numerically_correct":
                    branch_correct,

                "errors":
                    errors
            }
        )


    result[
        "correct_matched_branches"
    ] = correct_matched_branches


    result[
        "all_gt_branches_correct"
    ] = (
        len(
            correct_gt_indices
        )
        == ground_truth_count
    )


    # ---------------------------------------------
    # Extra predicted branches
    # ---------------------------------------------

    extra_predicted_branches = max(
        0,
        listed_count
        - len(matching)
    )


    result[
        "extra_predicted_branches"
    ] = extra_predicted_branches


    result[
        "no_extra_predicted_branches"
    ] = (
        extra_predicted_branches
        == 0
    )


    # ---------------------------------------------
    # Complete numerical configuration set
    # ---------------------------------------------

    result[
        "complete_set_correct"
    ] = (

        result[
            "configuration_count_correct"
        ]

        and result[
            "internal_consistency"
        ]

        and result[
            "all_gt_branches_correct"
        ]

        and result[
            "no_extra_predicted_branches"
        ]
    )


    # ---------------------------------------------
    # Independent physical validity
    # ---------------------------------------------

    input_parameters = ground_truth[
        "input_parameters"
    ]


    physical_flags = []


    for (
        predicted_index,
        configuration
    ) in enumerate(
        predicted_configurations
    ):

        physical_result = (
            physical_residual(
                configuration,
                input_parameters,
                mechanism_family
            )
        )


        physical_flags.append(
            physical_result[
                "physically_valid"
            ]
        )


        result[
            "physical_validity_details"
        ].append(
            {
                "predicted_index":
                    predicted_index,

                "maximum_residual_mm":
                    physical_result[
                        "maximum_residual_mm"
                    ],

                "normalized_residual":
                    physical_result[
                        "normalized_residual"
                    ],

                "physically_valid":
                    physical_result[
                        "physically_valid"
                    ]
            }
        )


    # Empty list is valid for a correctly
    # predicted infeasible mechanism.
    result[
        "all_reported_physically_valid"
    ] = all(
        physical_flags
    )


    # ---------------------------------------------
    # Main case-level outcomes
    # ---------------------------------------------

    result[
        "physical_exact_success"
    ] = (

        result[
            "complete_set_correct"
        ]

        and result[
            "all_reported_physically_valid"
        ]
    )


    result[
        "strict_task_success"
    ] = (

        result[
            "physical_exact_success"
        ]

        and result[
            "strict_format"
        ]

        and result[
            "internal_consistency"
        ]
    )


    return result

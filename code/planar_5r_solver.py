import math


def solve_planar_5r(
    L1,
    L2,
    L3,
    L4,
    L5,
    theta2_deg,
    theta5_deg
):

    # Convert input angles to radians
    theta2 = math.radians(theta2_deg)
    theta5 = math.radians(theta5_deg)

    # Fixed ground pivots
    A = (0.0, 0.0)
    D = (L1, 0.0)

    # Left crank endpoint
    B = (
        L2 * math.cos(theta2),
        L2 * math.sin(theta2)
    )

    # Right crank endpoint
    E = (
        L1 + L5 * math.cos(theta5),
        L5 * math.sin(theta5)
    )


    # Distance between the two circle centers
    dx = E[0] - B[0]
    dy = E[1] - B[1]

    distance_BE = math.hypot(
        dx,
        dy
    )

    # Numerical tolerance
    scale = max(
        abs(L1),
        abs(L2),
        abs(L3),
        abs(L4),
        abs(L5),
        1.0
    )

    linear_tol = 1e-10 * scale

    # Check circle-intersection condition
    outer_limit = L3 + L4
    inner_limit = abs(L3 - L4)

    if distance_BE <= linear_tol:

        if abs(L3 - L4) <= linear_tol:
            status = "degenerate_excluded"
        else:
            status = "infeasible"

        feasible = False
        n_branches = 0

    elif (
        distance_BE > outer_limit + linear_tol
        or
        distance_BE < inner_limit - linear_tol
    ):

        status = "infeasible"
        feasible = False
        n_branches = 0

    elif (
        abs(distance_BE - outer_limit) <= linear_tol
        or
        abs(distance_BE - inner_limit) <= linear_tol
    ):

        status = "tangent"
        feasible = True
        n_branches = 1

    else:

        status = "regular"
        feasible = True
        n_branches = 2


    # -------------------------------------------------
    # Circle-circle intersection
    # -------------------------------------------------

    a = None
    P = None
    h = None
    C_candidates = []

    if feasible:

        # Unit vector from B toward E
        ex = dx / distance_BE
        ey = dy / distance_BE

        # Distance from B to the midpoint line
        a = (
            L3**2
            - L4**2
            + distance_BE**2
        ) / (2.0 * distance_BE)

        # Base point P
        Px = B[0] + a * ex
        Py = B[1] + a * ey

        P = (Px, Py)

        # Perpendicular distance to intersection points
        h_squared = L3**2 - a**2

        squared_tol = 1e-10 * scale**2

        # Protect against tiny floating-point negatives
        if h_squared < 0 and abs(h_squared) <= squared_tol:
            h_squared = 0.0

        h = math.sqrt(
            max(0.0, h_squared)
        )

        if status == "tangent":

            C_candidates = [
                (Px, Py)
            ]

        else:

            C1 = (
                Px - h * ey,
                Py + h * ex
            )

            C2 = (
                Px + h * ey,
                Py - h * ex
            )

            C_candidates = [
                C1,
                C2
            ]


    # -------------------------------------------------
    # Process each assembly configuration
    # -------------------------------------------------

    solutions = []

    for C in C_candidates:

        Cx, Cy = C

        # Left coupler angle BC
        theta3 = math.atan2(
            Cy - B[1],
            Cx - B[0]
        )

        theta3_deg = (
            math.degrees(theta3)
            % 360.0
        )

        # Right coupler angle EC
        theta4 = math.atan2(
            Cy - E[1],
            Cx - E[0]
        )

        theta4_deg = (
            math.degrees(theta4)
            % 360.0
        )

        # Branch signature
        cross_value = (
            (E[0] - B[0]) * (Cy - B[1])
            -
            (E[1] - B[1]) * (Cx - B[0])
        )

        branch_tol = 1e-10 * scale**2

        # Tangency means the two assembly branches have merged
        if status == "tangent":

            branch_signature = 0

        elif cross_value > branch_tol:

            branch_signature = +1

        elif cross_value < -branch_tol:

            branch_signature = -1

        else:

            branch_signature = 0
        # Branch-merging singularity score
        singularity_score = abs(
            math.sin(
                theta3 - theta4
            )
        )

        # Check BC length
        BC_length = math.hypot(
            Cx - B[0],
            Cy - B[1]
        )

        BC_error = abs(
            BC_length - L3
        )

        # Check EC length
        EC_length = math.hypot(
            Cx - E[0],
            Cy - E[1]
        )

        EC_error = abs(
            EC_length - L4
        )

        # Check left loop A -> B -> C
        left_residual_x = (
            L2 * math.cos(theta2)
            + L3 * math.cos(theta3)
            - Cx
        )

        left_residual_y = (
            L2 * math.sin(theta2)
            + L3 * math.sin(theta3)
            - Cy
        )

        left_loop_residual = math.hypot(
            left_residual_x,
            left_residual_y
        )

        # Check right loop D -> E -> C
        right_residual_x = (
            L1
            + L5 * math.cos(theta5)
            + L4 * math.cos(theta4)
            - Cx
        )

        right_residual_y = (
            L5 * math.sin(theta5)
            + L4 * math.sin(theta4)
            - Cy
        )

        right_loop_residual = math.hypot(
            right_residual_x,
            right_residual_y
        )

        validation_tol = 1e-9 * scale

        validation_passed = (
            BC_error <= validation_tol
            and EC_error <= validation_tol
            and left_loop_residual <= validation_tol
            and right_loop_residual <= validation_tol
        )

        solutions.append(
            {
                "C": C,
                "theta3_deg": theta3_deg,
                "theta4_deg": theta4_deg,
                "branch_signature": branch_signature,
                "singularity_score": singularity_score,
                "BC_error": BC_error,
                "EC_error": EC_error,
                "left_loop_residual": left_loop_residual,
                "right_loop_residual": right_loop_residual,
                "validation_passed": validation_passed
            }
        )

    return {
        "status": status,
        "A": A,
        "B": B,
        "D": D,
        "E": E,
        "distance_BE": distance_BE,
        "a": a,
        "P": P,
        "h": h,
        "feasible": feasible,
        "n_branches": n_branches,
        "C_candidates": C_candidates,
        "solutions": solutions
    }


if __name__ == "__main__":

    result = solve_planar_5r(
        L1=100,
        L2=40,
        L3=70,
        L4=70,
        L5=40,
        theta2_deg=60,
        theta5_deg=120
    )

    print(result)

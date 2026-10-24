import math


def solve_slider_crank(L2, L3, e, theta2_deg):

    # Convert input angle to radians
    theta2 = math.radians(theta2_deg)

    # Fixed pivot A
    A = (0.0, 0.0)

    # Crank endpoint B
    Bx = L2 * math.cos(theta2)
    By = L2 * math.sin(theta2)

    B = (Bx, By)

    # Distance from B to the slider line
    vertical_distance = e - By

    # Circle-line intersection quantity
    q_squared = L3**2 - vertical_distance**2

    # Numerical tolerance
    scale = max(
        abs(L2),
        abs(L3),
        abs(e),
        1.0
    )

    squared_tol = 1e-10 * scale**2

    # Determine physical feasibility
    if q_squared < -squared_tol:

        status = "infeasible"
        feasible = False
        n_branches = 0
        q = None

    elif abs(q_squared) <= squared_tol:

        status = "tangent"
        feasible = True
        n_branches = 1
        q = 0.0

    else:

        status = "regular"
        feasible = True
        n_branches = 2
        q = math.sqrt(q_squared)

    # -----------------------------------------------------
    # Generate all physically valid slider positions
    # -----------------------------------------------------

    C_candidates = []

    if n_branches == 1:

        C = (Bx, e)
        C_candidates.append(C)

    elif n_branches == 2:

        C1 = (Bx + q, e)
        C2 = (Bx - q, e)

        C_candidates.append(C1)
        C_candidates.append(C2)

    # -----------------------------------------------------
    # Calculate information for every configuration
    # -----------------------------------------------------

    solutions = []

    branch_tol = 1e-10 * scale

    for C in C_candidates:

        # Connecting-rod angle theta3
        theta3 = math.atan2(
            C[1] - B[1],
            C[0] - B[0]
        )

        # Convert to degrees between 0 and 360
        theta3_deg = math.degrees(theta3) % 360.0

                # Singularity score
        singularity_score = abs(
            math.cos(theta3)
        )

        # Check connecting-rod length BC
        BC = math.hypot(
            C[0] - B[0],
            C[1] - B[1]
        )

        BC_error = abs(BC - L3)
        # Check slider constraint y = e
        slider_error = abs(
            C[1] - e
        )

                # Loop-closure residual
        residual_x = (
            L2 * math.cos(theta2)
            + L3 * math.cos(theta3)
            - C[0]
        )

        residual_y = (
            L2 * math.sin(theta2)
            + L3 * math.sin(theta3)
            - e
        )

        loop_residual = math.hypot(
            residual_x,
            residual_y
        )

                # Final numerical validation
        validation_tol = 1e-9 * scale

        validation_passed = (
            BC_error <= validation_tol
            and slider_error <= validation_tol
            and loop_residual <= validation_tol
        )
        
        # Branch signature
        horizontal_difference = C[0] - B[0]

        if horizontal_difference > branch_tol:
            branch_signature = +1

        elif horizontal_difference < -branch_tol:
            branch_signature = -1

        else:
            branch_signature = 0

        solution = {
            "C": C,
            "theta3_deg": theta3_deg,
            "branch_signature": branch_signature,
            "singularity_score": singularity_score,
            "BC_error": BC_error,
            "slider_error": slider_error,
            "loop_residual": loop_residual,
            "validation_passed": validation_passed
        }
        solutions.append(solution)

    return {
    "status": status,
    "A": A,
    "B": B,
    "vertical_distance": vertical_distance,
    "q_squared": q_squared,
    "q": q,
    "feasible": feasible,
    "n_branches": n_branches,
    "C_candidates": C_candidates,
    "solutions": solutions
}


if __name__ == "__main__":

    result = solve_slider_crank(
        L2=40.0,
        L3=70.0,
        e=20.0,
        theta2_deg=60.0
)

    print("A =", result["A"])
    print("B =", result["B"])
    print("q squared =", result["q_squared"])
    print("q =", result["q"])
    print("Feasible =", result["feasible"])
    print("Number of configurations =", result["n_branches"])
    print("Status =", result["status"])
    print("C positions =", result["C_candidates"])
    print("\nSolutions:")

    for i, solution in enumerate(
        result["solutions"],
        start=1
    ):

        print("\nConfiguration", i)
        print("C =", solution["C"])
        print("theta3 =", solution["theta3_deg"], "degrees")
        print(
            "Branch signature =",
            solution["branch_signature"]
        )
        print(
            "Singularity score =",
            solution["singularity_score"]
        )
        print(
            "BC error =",
            solution["BC_error"]
        )
        print(
            "Slider error =",
            solution["slider_error"]
        )
        print(
            "Loop residual =",
            solution["loop_residual"]
        )
        print(
            "Validation passed =",
            solution["validation_passed"]
        )

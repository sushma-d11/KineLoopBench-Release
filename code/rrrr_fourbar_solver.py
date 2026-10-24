import math


def solve_fourbar(L1, L2, L3, L4, theta2_deg):

    # Convert input angle from degrees to radians
    theta2 = math.radians(theta2_deg)

    # Fixed pivots
    A = (0.0, 0.0)
    D = (L1, 0.0)

    # Position of point B
    Bx = L2 * math.cos(theta2)
    By = L2 * math.sin(theta2)

    B = (Bx, By)

    # Distance from B to D
    dx = D[0] - B[0]
    dy = D[1] - B[1]

    d = math.hypot(dx, dy)

    # Feasibility limits
    lower_limit = abs(L3 - L4)
    upper_limit = L3 + L4

    # Numerical tolerance for length comparisons
    scale = max(abs(L1), abs(L2), abs(L3), abs(L4), 1.0)
    length_tol = 1e-10 * scale

       # ---------------------------------------------------------
    # Determine feasibility and calculate exact configurations
    # ---------------------------------------------------------

    # Special case: B and D are essentially at the same point
    if d <= length_tol:

        if abs(L3 - L4) <= length_tol:
            return {
                "status": "degenerate_excluded",
                "A": A,
                "B": B,
                "D": D,
                "distance_BD": d,
                "feasible": None,
                "n_branches": None,
                "solutions": []
            }

        else:
            return {
                "status": "infeasible",
                "A": A,
                "B": B,
                "D": D,
                "distance_BD": d,
                "feasible": False,
                "n_branches": 0,
                "solutions": []
            }

    # No circle intersection
    if d > upper_limit + length_tol or d < lower_limit - length_tol:

        return {
            "status": "infeasible",
            "A": A,
            "B": B,
            "D": D,
            "distance_BD": d,
            "feasible": False,
            "n_branches": 0,
            "solutions": []
        }

    # Unit vector from B to D
    ex = dx / d
    ey = dy / d

    # Circle-intersection geometry
    a = (L3**2 - L4**2 + d**2) / (2.0 * d)

    h_squared = L3**2 - a**2

    # h_squared has units of length squared,
    # so use a squared-length tolerance
    squared_tol = 1e-10 * scale**2

    if abs(h_squared) <= squared_tol:
        h_squared = 0.0

    # This should normally already be caught by feasibility checking
    if h_squared < 0:

        return {
            "status": "infeasible",
            "A": A,
            "B": B,
            "D": D,
            "distance_BD": d,
            "feasible": False,
            "n_branches": 0,
            "solutions": []
        }

    h = math.sqrt(h_squared)

    # Midpoint P
    Px = B[0] + a * ex
    Py = B[1] + a * ey

    P = (Px, Py)

       # ---------------------------------------------------------
    # Generate candidate C positions
    # ---------------------------------------------------------

    C1 = (
        Px - h * ey,
        Py + h * ex
    )

    C_candidates = [C1]

    if h > length_tol:

        C2 = (
            Px + h * ey,
            Py - h * ex
        )

        C_candidates.append(C2)

    # ---------------------------------------------------------
    # Calculate information for every valid configuration
    # ---------------------------------------------------------

    solutions = []

    area_tol = 1e-10 * scale**2

    for C in C_candidates:

        # Coupler angle theta3
        theta3 = math.atan2(
            C[1] - B[1],
            C[0] - B[0]
        )

        # Rocker angle theta4
        theta4 = math.atan2(
            C[1] - D[1],
            C[0] - D[0]
        )

        # Convert to degrees between 0 and 360
        theta3_deg = math.degrees(theta3) % 360.0
        theta4_deg = math.degrees(theta4) % 360.0

        # Branch signature
        cross_value = (
            dx * (C[1] - B[1])
            - dy * (C[0] - B[0])
        )

        if cross_value > area_tol:
            branch_signature = +1

        elif cross_value < -area_tol:
            branch_signature = -1

        else:
            branch_signature = 0

        # Singularity score
        rho = abs(
            math.sin(theta3 - theta4)
        )
        # -----------------------------------------------------
        # Loop-closure validation
        # -----------------------------------------------------

        residual_x = (
            L2 * math.cos(theta2)
            + L3 * math.cos(theta3)
            - L1
            - L4 * math.cos(theta4)
        )

        residual_y = (
            L2 * math.sin(theta2)
            + L3 * math.sin(theta3)
            - L4 * math.sin(theta4)
        )

        loop_residual = math.hypot(
            residual_x,
            residual_y
        )

        # -----------------------------------------------------
        # Direct link-length validation
        # -----------------------------------------------------

        BC = math.hypot(
            C[0] - B[0],
            C[1] - B[1]
        )

        CD = math.hypot(
            C[0] - D[0],
            C[1] - D[1]
        )

        BC_error = abs(BC - L3)
        CD_error = abs(CD - L4)

        # Numerical tolerance for validating the solution
        validation_tol = 1e-9 * scale

        validation_passed = (
            loop_residual <= validation_tol
            and BC_error <= validation_tol
            and CD_error <= validation_tol
        )

        # Store everything belonging to this branch
        solution = {
            "C": C,
            "theta3_deg": theta3_deg,
            "theta4_deg": theta4_deg,
            "branch_signature": branch_signature,
            "singularity_score": rho,
            "loop_residual": loop_residual,
            "BC_error": BC_error,
            "CD_error": CD_error,
            "validation_passed": validation_passed
        }

        solutions.append(solution)

    # Number of configurations
    n_branches = len(solutions)

    if n_branches == 1:
        status = "tangent"
    else:
        status = "regular"

    return {
        "status": status,
        "A": A,
        "B": B,
        "D": D,
        "distance_BD": d,
        "P": P,
        "h": h,
        "feasible": True,
        "n_branches": n_branches,
        "solutions": solutions
    }

# ---------------------------------------------------------
# Test our function
# ---------------------------------------------------------

if __name__ == "__main__":
    
    result = solve_fourbar(
        L1=40,
        L2=40,
        L3=30,
        L4=30,
        theta2_deg=0
    )

    print("Point A =", result["A"])
    print("Point B =", result["B"])
    print("Point D =", result["D"])
    print("Distance BD =", result["distance_BD"])
    print("Feasible =", result["feasible"])
    print("Number of configurations =", result["n_branches"])
    print("Status =", result["status"])
    print("Point P =", result.get("P"))
    print("h =", result.get("h"))

    print("\nSolutions:")


    for i, solution in enumerate(result["solutions"], start=1):

        print("\nBranch", i)

        print(
            "C =",
            solution["C"]
        )

        print(
            "theta3 =",
            solution["theta3_deg"],
            "degrees"
        )

        print(
            "theta4 =",
            solution["theta4_deg"],
            "degrees"
        )

        print(
            "Branch signature =",
            solution["branch_signature"]
        )

        print(
            "Singularity score =",
            solution["singularity_score"]
        )
        print(
            "Loop residual =",
            solution["loop_residual"]
        )

        print(
            "BC error =",
            solution["BC_error"]
        )

        print(
            "CD error =",
            solution["CD_error"]
        )

        print(
            "Validation passed =",
            solution["validation_passed"]
        )

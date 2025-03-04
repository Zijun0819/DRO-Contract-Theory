import numpy as np


def subproblem_value(L, xi, alpha, gamma1, gamma2, gamma3, lam, hat_xi_n, theta):
    """
    Returns the objective inside the min_{xi} for a given xi:
      sum_{i} alpha_i ln(gamma2*xi + gamma3*L[i]) + lam * |xi - hat_xi_n|.
    """
    val_log = 0.0
    for i in range(len(L)):
        term1 = residul_calc(L, gamma1, theta, i)
        val_log += alpha[i] * (np.log(gamma2 * xi + gamma3 * L[i]) - term1)
    return val_log + lam * abs(xi - hat_xi_n)


def residul_calc(L, gamma1, theta, index):
    residul_value = gamma1*L[0] / theta[0]
    for i in range(1, index+1):
        residul_value += gamma1*(L[i] - L[i-1])/theta[i]

    return residul_value


def solve_xi_subproblem(L, lam, hat_xi_n, alpha, gamma1, gamma2, gamma3, xi_lower, xi_upper, theta):
    """
        Solves (approximately) the 1D subproblem:
           min_{xi in [xi_lower, xi_upper]} f(xi) = sum_{i=1}^I alpha[i]*ln(gamma2*xi + gamma3*L[i]) + lam*|xi-hat_xi|.
        Here we use grid search.
        Returns: (xi_star, f_value)
        Revise with 4 options and the final one is the bisection to find the optimal one in (xi_lower, hat_xi_n)
        1. 1st and 2nd options are the boundary points
        2. 3rd option is point hat_xi_n
        3. 4th option is the optimal value in (xi_lower, hat_xi_n), find via bisection
    """
    candidates = []
    vals = []

    # -- Evaluate subproblem at boundary xi_lower
    val_lo = subproblem_value(L, xi_lower, alpha, gamma1, gamma2, gamma3, lam, hat_xi_n, theta)
    candidates.append(xi_lower)
    vals.append(val_lo)

    # -- Evaluate subproblem at boundary xi_upper
    val_up = subproblem_value(L, xi_upper, alpha, gamma1, gamma2, gamma3, lam, hat_xi_n, theta)
    candidates.append(xi_upper)
    vals.append(val_up)

    # -- If hat_xi_n is in [xi_lower, xi_upper], evaluate there
    if xi_lower <= hat_xi_n <= xi_upper:
        val_hat = subproblem_value(L, hat_xi_n, alpha, gamma1, gamma2, gamma3, lam, hat_xi_n, theta)
        candidates.append(hat_xi_n)
        vals.append(val_hat)

    def f_xi(xi):
        # f_xi(xi) = sum_i alpha_i * gamma2/(gamma2 xi + gamma3 L[i])
        s = 0.0
        for i in range(len(L)):
            denom = gamma2*xi + gamma3*L[i]
            s += alpha[i]*gamma2/denom
        return s

    # Because f_xi(xi) is typically monotonically decreasing in xi (>= 0),
    # we check if it crosses lam in [xi_lower, hat_xi_n].
    if hat_xi_n >= xi_lower:
        f_lower = f_xi(xi_lower)
        f_hat = f_xi(hat_xi_n)
        # We want f_xi^* = lam. If lam is between f_hat and f_lower, there's a crossing.
        if (lam <= f_lower and lam >= f_hat):
            # do a bisection on [xi_lower, hat_xi_n]
            x_left, x_right = xi_lower, hat_xi_n
            for _ in range(50):  # up to e.g. 50 iters
                x_mid = 0.5*(x_left + x_right)
                f_mid = f_xi(x_mid)
                # If f_mid > lam, we need to move xi up (since f_xi is decreasing in xi).
                if f_mid > lam:
                    x_left = x_mid
                else:
                    x_right = x_mid
                if abs(f_mid - lam) < 1e-12:
                    break
            xi_star = 0.5*(x_left + x_right)
            val_star = subproblem_value(L, xi_star, alpha, gamma1, gamma2, gamma3, lam, hat_xi_n, theta)
            candidates.append(xi_star)
            vals.append(val_star)

    f_vals = np.array(vals)
    idx = np.argmin(f_vals)
    xi_star = candidates[idx]
    return xi_star, f_vals[idx]


def penalty(L, gamma1, theta):
    """
    penalty(L) = gamma1 * [L[0]/theta[0] + sum_{j=1..I-1} (L[j] - L[j-1]) / theta[j]]
    (Adjust indices as needed.)
    """
    val = gamma1 * (L[0] / theta[0])
    for j in range(1, len(L)):
        val += gamma1 * ((L[j] - L[j - 1]) / theta[j])
    return val

import numpy as np


def my_sqrt(z, branch=1, branch_cut_angle=0):
    """
    Complex square root with a variable branch cut angle and the option to choose the branch.

    Args:
        z (complex): complex number
        branch (int, optional): branch of the square root. 1 is the principal branch -1 is the 2nd branch. Defaults to 1.
        branch_cut_angle (int, optional): Angle of the branch cut with respect to the real axis. Defaults to 0.

    Returns:
        complex: Returns the square root of z on the chosen branch with the chosen branch cut.
    """

    if branch not in [-1, 1]:
        raise ValueError("Branch must be either 1 or -1")

    phi = np.angle(z)  # between -pi and pi with respect to the real axis
    phi = (
        np.mod(phi + branch_cut_angle, 2 * np.pi) - branch_cut_angle
    )  # redefine the angle to be between -branch cut angle and 2pi-branch cut angle
    r = branch * np.sqrt(np.abs(z))  # radius on the complex plain
    return r * np.exp(1j * phi / 2)


def kaellen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * x * z


def q(s, m1, m2, branch=1, branch_cut_angle=0):
    """
    Absolute 3-momentum of the daughter particles in the center-of-momentum frame of the mother particle.

    Args:
        s (_type_): mandelstam variable
        m1 (_type_): invariant mass of the first daughter particle
        m2 (_type_): invariant mass of the second daughter particle
        branch (int, optional): branch of the square root. 1 is the principal branch -1 is the 2nd branch. Defaults to 1.
        branch_cut_angle (int, optional): Angle of the branch cut with respect to the real axis. Defaults to 0.

    Returns:
        _type_: Returns the absolute 3-momentum of the daughter particles in the center-of-momentum frame of the mother particle.
    """

    kaellen_val = kaellen(s, m1**2, m2**2)
    return my_sqrt(kaellen_val, branch=branch, branch_cut_angle=branch_cut_angle) / (
        2 * np.sqrt(s)
    )


def rho(s, m1, m2, branch=1, branch_cut_angle=0):
    """
    Phase space factor for the decay of a particle into two daughter particles.

    Args:
        s (_type_): mandelstam variable
        m1 (_type_): invariant mass of the first daughter particle
        m2 (_type_): invariant mass of the second daughter particle
        branch (int, optional): branch of the square root. 1 is the principal branch -1 is the 2nd branch. Defaults to 1.
        branch_cut_angle (int, optional): Angle of the branch cut with respect to the real axis. Defaults to 0.

    Returns:
        _type_: Returns the phase space factor for the decay of a particle into two daughter particles.
    """
    q_val = q(s, m1, m2, branch=branch, branch_cut_angle=branch_cut_angle)
    return 1.0 / (16 * np.pi) * 2 * q_val / np.sqrt(s)


def B(s, m1, m2, L, q0, branch=1, branch_cut_angle=0):
    if L == 0:
        return 1.0
    else:
        q_val = q(s, m1, m2, branch=branch, branch_cut_angle=branch_cut_angle)
        x = q_val / q0
    if L == 1:
        B_val = x / np.sqrt(1 + x**2)
    elif L == 2:
        B_val = x**2 / np.sqrt(9 + 3 * x**2 + x**4)
    else:
        raise ValueError("L must be 1 or 2")
    return B_val


def cm(s, m1, m2, branch=1, branch_cut_angle=0):
    q_val = q(s, m1, m2, branch, branch_cut_angle)
    return (
        1
        / (16 * np.pi**2)
        * (
            2
            * q_val
            / np.sqrt(s)
            * np.log((m1**2 + m2**2 - s + 2 * np.sqrt(s) * q_val) / (2 * m1 * m2))
            - (m1**2 - m2**2) * (1 / s - 1 / (m1 + m2) ** 2) * np.log(m1 / m2)
        )
    )


def q_cm(s, m1, m2, branch=1, branch_cut_angle=0):
    return 8.0 * np.pi * np.sqrt(s) * np.imag(cm(s, m1, m2, branch, branch_cut_angle))


def B_cm(s, m1, m2, L, q0, branch=1, branch_cut_angle=0):
    if L == 0:
        return 1.0
    else:
        q_val = q_cm(s, m1, m2, branch=branch, branch_cut_angle=branch_cut_angle)
        x = q_val / q0
    if L == 1:
        B_val = x / np.sqrt(1 + x**2)
    elif L == 2:
        B_val = x**2 / np.sqrt(9 + 3 * x**2 + x**4)
    else:
        raise ValueError("L must be 1 or 2")
    B_val[B_val < 1e-10] = 0.0
    return B_val


def rho_dudek(s, m1, m2):
    if abs(s - (m1 + m2) ** 2) < 1e-10:
        return 0.0 + 0.0j
    return np.sqrt(1.0 - (m1 + m2) ** 2 / s) * np.sqrt(1.0 - (m1 - m2) ** 2 / s)


def rho_dudek_arr(s, m1, m2):
    rho_vals = np.zeros_like(s, dtype=complex)
    valid = np.abs(s - (m1 + m2) ** 2) >= 1e-10
    s_valid = s[valid]
    rho_vals[valid] = np.sqrt(1.0 - (m1 + m2) ** 2 / s_valid) * np.sqrt(
        1.0 - (m1 - m2) ** 2 / s_valid
    )
    return rho_vals


def cm_dudek(s, m1, m2, branch=1):
    if abs(s - (m1 + m2) ** 2) < 1e-10:
        return 0.0 + 0.0j
    rho_val = rho_dudek(s, m1, m2)
    ep = 1.0 - (m1 + m2) ** 2 / s
    if branch == 1:
        return rho_val / np.pi * np.log(
            (ep + rho_val) / (ep - rho_val)
        ) - ep / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
    if branch == -1:
        return np.conjugate(
            rho_val / np.pi * np.log((ep + rho_val) / (ep - rho_val))
            + ep / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
        )


def cm_dudek_arr(s, m1, m2, branch=1):
    cm_vals = np.zeros_like(s, dtype=complex)
    valid = np.abs(s - (m1 + m2) ** 2) >= 1e-10
    s_valid = s[valid]
    rho_val = rho_dudek_arr(s_valid, m1, m2)
    ep = 1.0 - (m1 + m2) ** 2 / s_valid
    if branch == 1:
        cm_vals[valid] = rho_val / np.pi * np.log(
            (ep + rho_val) / (ep - rho_val)
        ) - ep / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
    if branch == -1:
        cm_vals[valid] = np.conjugate(
            rho_val / np.pi * np.log((ep + rho_val) / (ep - rho_val))
            + ep / np.pi * (m2 - m1) / (m1 + m2) * np.log(m2 / m1)
        )
    return cm_vals


def amp(s, m1, m2, L, q0, g, beta, m0, c, b, branch=1, branch_cut_angle=0):
    n_val = B(s, m1, m2, L, q0, branch=branch, branch_cut_angle=branch_cut_angle)
    rho_val = rho(s, m1, m2, branch=branch, branch_cut_angle=branch_cut_angle)
    K_mat_elem = n_val**2 * (g**2 / (m0**2 - s) + c)
    P_vec_elem = (beta * g / (m0**2 - s) + b) * n_val
    return P_vec_elem / (1 - K_mat_elem * 1.0j * rho_val)


def amp2(s, m1, m2, L, q0, g, beta, m0, c, b, branch=1, branch_cut_angle=0):
    n_val = B_cm(s, m1, m2, L, q0, branch=branch, branch_cut_angle=branch_cut_angle)
    cm_val = cm(s, m1, m2, branch=branch, branch_cut_angle=branch_cut_angle)
    K_mat_elem = n_val**2 * (g**2 / (m0**2 - s) + c)
    P_vec_elem = (beta * g / (m0**2 - s) + b) * n_val
    return P_vec_elem / (1 - K_mat_elem * cm_val)


def amp3(s, m1, m2, L, q0, g, beta, m0, c, b, branch=1, branch_cut_angle=0):
    n_val = B_cm(s, m1, m2, L, q0, branch=branch, branch_cut_angle=branch_cut_angle)
    cm_val = cm_dudek_arr(s, m1, m2, branch=branch)
    K_mat_elem = n_val**2 * (g**2 / (m0**2 - s) + c)
    P_vec_elem = (beta * g / (m0**2 - s) + b) * n_val
    return P_vec_elem / (1 + K_mat_elem * 1.0 / (16 * np.pi) * cm_val)


def K_matrix(s, mR, gR, b, R, C, B, s0, s_norm):
    s_hat = s / s0 - 1.0
    K_mat = np.zeros((C, C), dtype=complex)
    for i in range(C):
        for j in range(i, C):
            bkg = 0.0 + 0.0j
            for bg in range(B + 1):
                bkg += b[i, j, bg] * s_hat**bg
            res = 0.0 + 0.0j
            for r in range(R):
                res += gR[i, r] * gR[j, r] / (mR[r] ** 2 - s) + bkg
            K_mat[i, j] = (s - s0) / s_norm * res
            K_mat[j, i] = K_mat[i, j]
    return K_mat


def phsp_matrix(s, mC, C, branch):
    phsp_mat = np.zeros((C, C), dtype=complex)
    for c in range(C):
        phsp_mat[c, c] = cm_dudek(s, mC[c, 0], mC[c, 1], branch[c])
    return phsp_mat


def T_matrix(s, mR, mC, gR, b, R, C, B, s0, s_norm, branch=None):
    if branch is None:
        branch = np.ones(C)
    K_mat = K_matrix(s, mR, gR, b, R, C, B, s0, s_norm)
    phsp_mat = phsp_matrix(s, mC, C, branch)
    if C > 1:
        denom = np.identity(C) + K_mat @ phsp_mat
        denom_inv = np.linalg.inv(denom)
        return denom_inv @ K_mat
    else:
        return K_mat / (1 + K_mat * phsp_mat[0, 0])


def P_vector(s, mR, gR, beta, b, R, C, B):
    P_vec = np.zeros(C, dtype=complex)
    for i in range(C):
        bkg = 0.0 + 0.0j
        for bg in range(B + 1):
            bkg += b[i, bg] * s**bg
        res = 0.0 + 0.0j
        for r in range(R):
            res += gR[i, r] * beta[r] / (mR[r] ** 2 - s) + bkg
        P_vec[i] = res
    return P_vec


def F_vector(
    s,
    mR,
    mC,
    gR,
    beta,
    b_K_mat,
    b_P_vec,
    R,
    C,
    B_K_mat,
    B_P_vec,
    s0,
    s_norm,
    branch=None,
):
    K_mat = K_matrix(s, mR, gR, b_K_mat, R, C, B_K_mat, s0, s_norm)
    P_vec = P_vector(s, mR, gR, beta, b_P_vec, R, C, B_P_vec)
    if branch is None:
        branch = np.ones(C)
    phsp_mat = phsp_matrix(s, mC, C, branch)
    if C > 1:
        denom = np.identity(C) + K_mat @ phsp_mat
        denom_inv = np.linalg.inv(denom)
        return denom_inv @ P_vec
    else:
        return P_vec / (1 + K_mat * phsp_mat[0, 0])

# -*- coding: utf-8 -*-

# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
================================================================================
Generalised Skyrme Equation (GSE) — numerical hedgehog solver (B = 1)
================================================================================

Goal of this verifier
---------------------
This script solves the dimensionless radial hedgehog ODE arising from the
SU(3) generalised Skyrme equation (GSE) in the baryonic sector B = 1,
checks Derrick’s virial balance (E₂ ≈ E₄), and evaluates the dimensionless
functionals that feed into the calibration in the paper:
  (i)  E_stat  — the static energy (dimensionless),
  (ii) Λ       — the (dimensionless) isospin moment of inertia integral,
  (iii) E₂, E₄ — quadratic and quartic contributions (virial test).

All equations below use Unicode-only maths; no LaTeX markup.

Field content, Lagrangian, and GSE (context)
--------------------------------------------
• Field: U(x) ∈ SU(3); left-invariant current L_μ = U† ∂_μ U ∈ su(3).
• Lagrangian:
    𝓛 = 𝓛₂ + 𝓛₄
       = (f_π² / 4) Tr(L_μ L^μ) + (1 / 32 e²) Tr([L_μ, L_ν][L^μ, L^ν]).
• Generalised Skyrme Equation (GSE) in compact SU(3)-covariant form:
    D_μ J^μ = 0,
  with
    J^μ = ∂𝓛/∂L_μ = (f_π² / 2) L^μ + (1 / 8 e²) [L_ν, [L^ν, L^μ]],
    D_μ(·) = ∂_μ(·) + [L_μ, ·].

Hedgehog ansatz and dimensionless radial ODE
--------------------------------------------
Adopt the spherically symmetric SU(2) ⊂ SU(3) hedgehog ansatz and scale the
radius to x = e f_π r. Writing F = F(x), the dimensionless ODE reads:
  ( 1 + (2 sin²F) / x² ) F″ + (2/x) F′
  − (sin(2F)/x²) [ 1 + (F′)² − (sin²F)/x² ] = 0.

In this code we denote the independent variable by r (already dimensionless)
to keep the notation minimal. The equation is implemented exactly as:
  A(r,F) F″ + (2/r) F′ − (sin(2F)/r²) [ 1 + (F′)² − (sin²F)/r² ] = 0,
with A(r,F) = 1 + 2 sin²F / r².

Boundary conditions and far-field behaviour
-------------------------------------------
B = 1 hedgehog boundary conditions:
  F(0) = π,
  F(r) ~ C / r²  as r → ∞.
The latter implies the Robin condition at a finite truncation R:
  r F′(R) + 2 F(R) = 0.

Energy functionals and virial decomposition
-------------------------------------------
Split the static energy (dimensionless) into “quadratic” (E₂) and “quartic”
(E₄) parts under the hedgehog ansatz:
  E_stat = 4π ∫ [ ½( F′² + 2 sin²F / r² )
                + ½( (sin²F / r²) F′² + (sin⁴F) / (2 r⁴) ) ] r² dr.

Define:
  E₂ = 4π ∫ ½( F′² + 2 sin²F / r² ) r² dr,
  E₄ = 4π ∫ ½( (sin²F / r²) F′² + (sin⁴F) / (2 r⁴) ) r² dr,
so that E_stat = E₂ + E₄.

Isospin inertia (dimensionless Λ-integral, proportional to I_𝓘):
  Λ = (8π/3) ∫ r² sin²F [ 1 + ( F′² + sin²F / r² ) ] dr.

Derrick’s scaling and virial balance
------------------------------------
Under the scale transform r ↦ r/λ (i.e. F_λ(r) = F(λ r)), one has:
  E₂(λ) = λ E₂,     E₄(λ) = λ⁻¹ E₄.
The stationary point is at λ* = √(E₄/E₂), where E₂(λ*) = E₄(λ*),
and E_stat(λ*) = 2 √(E₂ E₄).

Λ scales as:
  Λ(λ) = (8π/3) [ λ⁻³ I0 + λ⁻¹ I1 ],
with
  I0 = ∫ r² sin²F dr,
  I1 = ∫ r² sin²F ( F′² + sin²F / r² ) dr.

Tail corrections (finite box at r = R)
--------------------------------------
For F(r) ~ C / r², the missing tail contributes:
  ΔE₂_tail = 4π C² / R³,   ΔE₄_tail ≈ 0,
  ΔΛ_tail  = (8π/3) C² / R.
We estimate C on the boundary in two ways and average:
  C1 = F(R) R²,     C2 = −½ R³ F′(R),     C = (C1 + C2)/2.

Success criteria for verification
---------------------------------
1) Newton converges (‖residual‖_∞ ≪ 1e−8), F is monotonic, F′(R) ≈ −2 F(R)/R.
2) After analytic Derrick scaling: E₂*/E₄* ≈ 1 and Λ* in a sensible range.
3) Stability with respect to R, N; small tail corrections.

Interpretation
--------------
• Virial balance and convergence ⇒ a genuine minimiser (stable soliton).
• Λ gives the right order for the moment of inertia used in collective SU(2)
  quantisation and hence in the N–Δ splitting.
• With calibrated (f_π, e) from the baryon sector, one can predict spectra.

The code below
--------------
The solver uses:
• a robust shooting (RK4 + secant) to generate a good initial profile,
• a global Newton method (finite-difference Jacobian, backtracking line search),
• post-processing for E₂, E₄, E_stat, Λ, tail estimates, and Derrick scaling,
• a crude radial small-oscillation proxy to check positivity of low modes,
• simple plots for F(r) and F′(r).

"""
# ============================================================

import numpy as np
import math
import matplotlib.pyplot as plt


# ---------------------------
# Grid and derivatives
# ---------------------------
def build_grid(R=18.0, N=2000):
    """
    Build a uniform dimensionless radial grid r ∈ [0, R] with N nodes.

    Parameters
    ----------
    R : float
        Truncation radius (dimensionless x = e f_π r in the paper’s notation).
        Acts as the far-field cut-off where a Robin boundary condition is enforced.
    N : int
        Number of grid points (including endpoints).

    Returns
    -------
    numpy.ndarray
        Array of r-values of shape (N,), linearly spaced from 0 to R.
    """
    return np.linspace(0.0, R, N)


def derivs(F, r):
    """
    Compute first and second radial derivatives of the profile F(r) by
    centred finite differences (numpy.gradient, edge_order=2).

    Parameters
    ----------
    F : numpy.ndarray
        Profile samples F(r) on the grid r.
    r : numpy.ndarray
        Radial grid as produced by build_grid().

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple (F′(r), F″(r)).
    """
    Fr  = np.gradient(F, r, edge_order=2)
    Frr = np.gradient(Fr, r, edge_order=2)
    return Fr, Frr


# ---------------------------
# ODE residual + second-order Robin BC at r = R
# ---------------------------
def hedgehog_residual(F, r):
    """
    Compute the discrete residual R[F](r) of the dimensionless hedgehog ODE:
      A F″ + (2/r) F′ − (sin(2F)/r²) [ 1 + (F′)² − sin²F / r² ] = 0,
    where A = 1 + 2 sin²F / r².

    Boundary conditions enforced as residuals:
      • At r = 0:   F(0) − π = 0  (B = 1 hedgehog condition),
      • At r = R:   r F′(R) + 2 F(R) = 0  (Robin BC from F ~ C / r²).

    Notes
    -----
    • To avoid division by zero, r is clipped below at 1e−12 in algebraic terms.
    • The right-end derivative F′(R) is evaluated with a 2nd-order one-sided
      stencil: F′(R) ≈ (3 F_N − 4 F_{N−1} + F_{N−2}) / (2 Δr).

    Parameters
    ----------
    F : numpy.ndarray
        Profile values.
    r : numpy.ndarray
        Radial grid.

    Returns
    -------
    numpy.ndarray
        Residual vector with the same shape as F.
    """
    Fr, Frr = derivs(F, r)
    rs = np.maximum(r, 1e-12)
    s  = np.sin(F)
    s2 = np.sin(2.0*F)

    # A(r,F) = 1 + 2 sin²F / r²
    A = 1.0 + 2.0*(s*s)/(rs*rs)

    # ODE residual in the interior
    res = A*Frr + (2.0/rs)*Fr - (s2/(rs*rs))*(1.0 + Fr*Fr - (s*s)/(rs*rs))

    # BC at r=0: enforce F(0)=π by setting the residual directly
    res = res.copy()
    res[0] = F[0] - math.pi

    # BC at r=R: Robin r F′(R) + 2 F(R) = 0, with 2nd-order derivative
    dr = r[-1] - r[-2]
    Fr_R = (3*F[-1] - 4*F[-2] + F[-3]) / (2*dr)
    res[-1] = r[-1]*Fr_R + 2.0*F[-1]
    return res


# ---------------------------
# Numerical Jacobian (five-diagonal fill pattern)
# ---------------------------
def numerical_jacobian(F, r, eps_fd=1e-6):
    """
    Assemble a numerical finite-difference Jacobian J ≈ ∂R/∂F, column by column,
    populating a narrow band (up to ±2 neighbours) for efficiency.

    For each column j (perturbing F_j by d), compute:
      J[:, j] ≈ ( R[F + d e_j] − R[F] ) / d,  with  d = eps_fd (1 + |F_j|).

    Parameters
    ----------
    F : numpy.ndarray
        Current profile.
    r : numpy.ndarray
        Radial grid.
    eps_fd : float
        Relative finite-difference step magnitude.

    Returns
    -------
    numpy.ndarray
        Dense Jacobian matrix J of shape (N, N) (banded content).
    """
    N = len(F)
    J = np.zeros((N, N))
    base = hedgehog_residual(F, r)
    for i in range(N):
        for j in (i-2, i-1, i, i+1, i+2):
            if 0 <= j < N:
                Fj = F.copy()
                d = eps_fd*(1.0 + abs(F[j]))
                Fj[j] += d
                resj = hedgehog_residual(Fj, r)
                J[:, j] = (resj - base) / d
    return J


# ---------------------------
# Newton solver with backtracking line search
# ---------------------------
def newton_solve(F0, r, tol=1e-10, max_iter=30):
    """
    Solve R[F] = 0 by damped Newton’s method with backtracking.

    Algorithm
    ---------
    Given F, compute residual R and Jacobian J ≈ ∂R/∂F:
      J Δ = −R,  F_try = F + α Δ,
    with a backtracking line search on α ∈ {1, 1/2, 1/4, …} to reduce ‖R‖_∞.

    Stopping:
      • success if ‖R‖_∞ < tol,
      • failure if backtracking cannot improve or max_iter reached.

    Parameters
    ----------
    F0 : numpy.ndarray
        Initial profile (good initialisation matters; shooting supplies it).
    r : numpy.ndarray
        Radial grid.
    tol : float
        Infinity-norm tolerance on the residual.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    (numpy.ndarray, bool, int, float)
        Tuple (F, ok, iters, resnorm):
          F       — converged profile (or last iterate if failed),
          ok      — True if converged, else False,
          iters   — number of Newton iterations performed,
          resnorm — final ‖R‖_∞.
    """
    F = F0.copy()
    for it in range(max_iter):
        Rv = hedgehog_residual(F, r)
        nrm = np.linalg.norm(Rv, np.inf)
        if nrm < tol:
            return F, True, it, nrm
        J = numerical_jacobian(F, r, eps_fd=1e-6)
        try:
            delta = np.linalg.solve(J, -Rv)
        except np.linalg.LinAlgError:
            # Fallback to least squares if J is ill-conditioned
            delta, *_ = np.linalg.lstsq(J, -Rv, rcond=None)
        alpha, best = 1.0, nrm
        while alpha > 1e-4:
            F_try = F + alpha*delta
            nrm_try = np.linalg.norm(hedgehog_residual(F_try, r), np.inf)
            if nrm_try < best:
                F, best = F_try, nrm_try
                break
            alpha *= 0.5
        else:
            # backtracking exhausted
            return F, False, it+1, best
    return F, False, max_iter, np.linalg.norm(hedgehog_residual(F, r), np.inf)


# ---------------------------
# Shooting (RK4 + secant) — good initial guess for Newton
# ---------------------------
def Fpp(r, F, G):
    """
    Second derivative F″ expressed explicitly from the ODE:
      A F″ + (2/r) G − (sin(2F)/r²) [ 1 + G² − sin²F / r² ] = 0,
    with A = 1 + 2 sin²F / r² and G = F′.
    Thus:
      F″ = { (sin(2F)/r²) [ 1 + G² − sin²F / r² ] − (2/r) G } / A.

    To avoid r = 0, the function clips r as rr = max(r, 1e−12).

    Parameters
    ----------
    r : float
        Radius at which F″ is evaluated.
    F : float
        Profile F(r).
    G : float
        First derivative F′(r).

    Returns
    -------
    float
        The value of F″(r) from the ODE.
    """
    rr = r if r>1e-12 else 1e-12
    s = math.sin(F); s2 = math.sin(2.0*F)
    denom = 1.0 + 2.0*(s*s)/(rr*rr)
    num   = (s2/(rr*rr))*(1.0 + G*G - (s*s)/(rr*rr)) - (2.0/rr)*G
    return num/denom


def integrate_profile(R=15.0, N=2000, a=1.2):
    """
    Integrate the first-order system:
      F′ = G,
      G′ = F″(r, F, G)  (from Fpp),
    by classical RK4 from r = r0 to r = R, with initial conditions enforcing
    F(0) = π via the small-r expansion F(r) ≈ π − a r, G(r) ≈ −a.

    Parameters
    ----------
    R : float
        Integration end radius (must be ≤ the final grid’s R used later).
    N : int
        Number of RK steps (N+1 samples will be produced).
    a : float
        Initial slope parameter in the linearised core: F ≈ π − a r.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple (r, F) containing the integration grid r and the profile F(r).
        The last value F(R) will be used by the secant method in shooting.
    """
    r0 = 1e-6
    dr = (R-r0)/N
    r = np.linspace(r0, R, N+1)
    F = np.empty_like(r); G = np.empty_like(r)
    F[0] = math.pi - a*r0; G[0] = -a
    for i in range(N):
        ri, Fi, Gi = r[i], F[i], G[i]
        # RK4 stages
        k1F, k1G = Gi, Fpp(ri,Fi,Gi)
        rm = ri + 0.5*dr
        Fm, Gm = Fi + 0.5*dr*k1F, Gi + 0.5*dr*k1G
        k2F, k2G = Gm, Fpp(rm, Fm, Gm)
        Fm2, Gm2 = Fi + 0.5*dr*k2F, Gi + 0.5*dr*k2G
        k3F, k3G = Gm2, Fpp(rm, Fm2, Gm2)
        re = ri + dr
        Fe, Ge = Fi + dr*k3F, Gi + dr*k3G
        k4F, k4G = Ge, Fpp(re, Fe, Ge)
        F[i+1] = Fi + (dr/6.0)*(k1F + 2*k2F + 2*k3F + k4F)
        G[i+1] = Gi + (dr/6.0)*(k1G + 2*k2G + 2*k3G + k4G)
    return r, F


def shoot_for_boundary(R=15.0, N=2200, a0=0.6, a1=2.0, tol=1e-8, itmax=40):
    """
    Secant-based shooting to satisfy the right-end boundary condition F(R) ≈ 0
    (consistent with the Robin BC imposed later on the Newton grid).

    Procedure
    ---------
    1) Integrate_profile(R, N, a) for two trial slopes a0, a1 at the origin.
    2) Measure the terminal mismatch f(a) = F(R; a).
    3) Update the slope by the secant method:
         a_{new} = a_c − f(a_c) (a_c − a_p) / ( f(a_c) − f(a_p) ).
    4) Stop if |f(a)| < tol or after itmax iterations.

    Parameters
    ----------
    R : float
        Shooting integration radius.
    N : int
        Number of RK4 steps (dense integration).
    a0, a1 : float
        Initial guesses for the core slope a.
    tol : float
        Tolerance on |F(R)|.
    itmax : int
        Maximum secant iterations.

    Returns
    -------
    (float, numpy.ndarray, numpy.ndarray, bool)
        Tuple (a_opt, r, F, ok):
          a_opt — best slope found,
          r, F  — last integrated profile,
          ok    — True if |F(R)| < tol.
    """
    r, F = integrate_profile(R, N, a0); f0 = F[-1]
    r, F = integrate_profile(R, N, a1); f1 = F[-1]
    ap, fp, ac, fc = a0, f0, a1, f1
    for _ in range(itmax):
        if abs(fc-fp) < 1e-14: break
        an = ac - fc*(ac-ap)/(fc-fp)
        r, F = integrate_profile(R, N, an); fn = F[-1]
        if abs(fn) < tol:
            return an, r, F, True
        ap, fp, ac, fc = ac, fc, an, fn
    return ac, r, F, abs(fc)<tol


# ---------------------------
# Energies / Λ and tail corrections
# ---------------------------
def estat_lambda_split(r, F):
    """
    Compute the dimensionless energies and Λ-decomposition on the given grid.

    Definitions (dimensionless):
      E₂ density: ½( F′² + 2 sin²F / r² ),
      E₄ density: ½( (sin²F / r²) F′² + ½ sin⁴F / r⁴ ),
      E₂  = 4π ∫ (E₂ density) r² dr,
      E₄  = 4π ∫ (E₄ density) r² dr,
      E_stat = E₂ + E₄.

    Λ decomposition:
      Λ   = (8π/3) ( I0 + I1 ),
      I0  = ∫ r² sin²F dr,
      I1  = ∫ r² sin²F ( F′² + sin²F / r² ) dr.

    Parameters
    ----------
    r : numpy.ndarray
        Radial grid.
    F : numpy.ndarray
        Profile on r.

    Returns
    -------
    (float, float, float, float, float, float)
        Tuple (E_stat, E₂, E₄, Λ, I0, I1).
    """
    Fr, _ = derivs(F, r)
    rs = np.maximum(r, 1e-12)
    s  = np.sin(F); s2 = s*s

    # E₂ and E₄ densities
    E2_d = 0.5*(Fr*Fr + 2.0*s2/(rs*rs))
    E4_d = 0.5*( (s2*Fr*Fr)/(rs*rs) + 0.5*(s2*s2)/(rs**4) )

    E2 = 4.0*math.pi*np.trapezoid(E2_d*(rs**2), rs)
    E4 = 4.0*math.pi*np.trapezoid(E4_d*(rs**2), rs)
    Estat = E2 + E4

    # Λ = (8π/3)(I0 + I1)
    I0 = np.trapezoid( (rs**2)*s2, rs )
    I1 = np.trapezoid( (rs**2)*s2*(Fr*Fr + s2/(rs*rs)), rs )
    Lambda = (8.0*math.pi/3.0)*(I0 + I1)
    return Estat, E2, E4, Lambda, I0, I1


def tail_constants(r, F):
    """
    Estimate the far-field constant C from the truncation boundary r = R,
    using two equivalent asymptotic formulae for F ~ C / r²:
      C1 = F(R) R²,
      C2 = −½ R³ F′(R),
    and return their average C = (C1 + C2)/2 together with diagnostics.

    Parameters
    ----------
    r : numpy.ndarray
        Radial grid.
    F : numpy.ndarray
        Profile.

    Returns
    -------
    (float, float, float, float)
        Tuple (C, C1, C2, F′(R)).
    """
    dr = r[-1] - r[-2]
    Fr_R = (3*F[-1] - 4*F[-2] + F[-3])/(2*dr)
    C1 = F[-1]*(r[-1]**2)
    C2 = -0.5*(r[-1]**3)*Fr_R
    return 0.5*(C1+C2), C1, C2, Fr_R


def tail_corrections(R, C):
    """
    Tail corrections for a finite box at r = R when F ~ C / r²:
      ΔE₂_tail = 4π C² / R³,    ΔE₄_tail ≈ 0,
      ΔΛ_tail  = (8π/3) C² / R.

    Parameters
    ----------
    R : float
        Truncation radius (last grid point).
    C : float
        Asymptotic constant from tail_constants().

    Returns
    -------
    (float, float, float)
        Tuple (ΔE₂_tail, ΔE₄_tail, ΔΛ_tail).
    """
    dE2 = 4.0*math.pi*(C*C)/(R**3)            # ΔE2_tail
    dE4 = 0.0                                  # ≈ 0
    dLam = (8.0*math.pi/3.0)*(C*C)/R           # ΔΛ_tail
    return dE2, dE4, dLam


# ---------------------------
# Derrick analytic scaling (integral rescaling, no profile resampling)
# ---------------------------
def derrick_scale(E2, E4, I0, I1, tail_dE2=0.0, tail_dLam=0.0):
    """
    Apply Derrick’s analytic scaling to bring E₂ and E₄ into balance.

    Theory
    ------
      λ* = √(E₄/E₂),
      E₂(λ*) = λ* E₂,   E₄(λ*) = E₄ / λ*,
      E_stat(λ*) = E₂(λ*) + E₄(λ*) = 2 √(E₂ E₄).

    Λ rescales as:
      Λ(λ) = (8π/3)( λ⁻³ I0 + λ⁻¹ I1 ).

    Tail corrections:
      ΔE₂_tail scales ∝ λ¹ (since it belongs to the quadratic part’s tail),
      ΔΛ_tail  scales ∝ λ⁻³ (dominated by I0-type tail).

    Parameters
    ----------
    E2, E4 : float
        Quadratic and quartic energy contributions (raw, before scaling).
    I0, I1 : float
        Λ-decomposition integrals as in estat_lambda_split().
    tail_dE2 : float
        Tail correction for E₂ at λ = 1; scaled as λ¹.
    tail_dLam : float
        Tail correction for Λ at λ = 1; scaled as λ⁻³.

    Returns
    -------
    (float, float, float, float, float)
        Tuple (λ*, E_stat*, Λ*, E_stat*+tail, Λ*+tail).
    """
    # λ* = sqrt(E4/E2)
    lam = math.sqrt(E4/E2)
    # scaled E2 and E4
    E2s = lam*E2
    E4s = E4/lam
    Estat_s = E2s + E4s           # == 2*sqrt(E2*E4)

    # Λ(λ) = (8π/3)( λ^{-3} I0 + λ^{-1} I1 )
    Lambda_s = (8.0*math.pi/3.0)*( (I0/(lam**3)) + (I1/lam) )

    # Tail scaling: ΔE2_tail ~ λ^1,  ΔΛ_tail ~ λ^{-3}
    Estat_s_corr = Estat_s + lam*tail_dE2
    Lambda_s_corr = Lambda_s + (tail_dLam/(lam**3))
    return lam, Estat_s, Lambda_s, Estat_s_corr, Lambda_s_corr


# ---------------------------
# Crude small-oscillation proxy (order-of-magnitude of ω)
# ---------------------------
def vib_eigs(r, F, k=3):
    """
    Construct a simple symmetric tridiagonal operator as a proxy for the
    radial fluctuation operator around F(r), and return the smallest k
    positive frequencies ω ≈ sqrt(eigenvalues).

    This is a diagnostic, not a rigorous stability analysis; it uses:
      • A(r) = 1 + 2 sin²F / r² as a position-dependent stiffness,
      • A positive-definite proxy potential V(r) ≈ 2 cos²F / r² + 2 sin²F / r⁴,
      • Dirichlet anchors at r = 0 and r = R (main[0] = main[-1] = 1).

    Parameters
    ----------
    r : numpy.ndarray
        Radial grid.
    F : numpy.ndarray
        Profile.
    k : int
        Number of lowest modes to return.

    Returns
    -------
    numpy.ndarray
        Array of the k smallest positive ω-values (dimensionless).
    """
    dr = r[1]-r[0]
    rs = np.maximum(r, 1e-12)
    s  = np.sin(F); c=np.cos(F)
    A = 1.0 + 2.0*(s*s)/(rs*rs)
    V = (2.0*(c*c)/(rs*rs)) + (s*s)*(2.0/(rs**4))  # positive proxy
    N = len(r)
    main = np.zeros(N); off = np.zeros(N-1)
    Ahalf = 0.5*(A[1:]+A[:-1])
    main[1:-1] = (Ahalf[1:]+Ahalf[:-1])/(dr*dr) + V[1:-1]
    off[:-1] = -Ahalf[:-1]/(dr*dr)
    off[1:]  = -Ahalf[1: ]/(dr*dr)
    main[0] = 1.0; main[-1] = 1.0
    H = np.zeros((N,N))
    idx = np.arange(N)
    H[idx,idx] = main
    H[idx[:-1], idx[1:]] = off
    H[idx[1:],  idx[:-1]] = off
    w2, _ = np.linalg.eigh(H)
    w2 = np.sort(w2); w2 = w2[w2>1e-10]
    return np.sqrt(w2[:k])


# ---------------------------
# Execution and reporting
# ---------------------------
if __name__ == "__main__":
    # Domain and resolution:
    # R — truncation radius for the ODE (dimensionless), N — grid points later.
    R = 18.0
    N = 2000

    # 1) Shooting to obtain a good initial profile for Newton
    a_opt, r_sh, F_sh, ok_sh = shoot_for_boundary(R=R, N=2200)
    if not ok_sh:
        print("[warn] shooting did not hit tol; using best secant iterate.")

    # 2) Newton solver on the final grid
    r = build_grid(R=R, N=N)
    F0 = np.interp(r, r_sh, F_sh)  # interpolate shooting profile onto final grid
    F, ok, iters, resnorm = newton_solve(F0, r, tol=1e-10, max_iter=30)
    print(f"Newton converged: {ok}, iters={iters}, ||R||_inf={resnorm:.3e}")

    # 3) Raw integrals and Λ decomposition
    Estat, E2, E4, Lambda, I0, I1 = estat_lambda_split(r, F)
    print(f"E_stat^(0) raw ≈ {Estat:.6f},  Λ^(0) raw ≈ {Lambda:.6f}")
    print(f"E2={E2:.6f}, E4={E4:.6f},  E2/E4 = {E2/E4:.6f}")

    # 4) Tail estimate and corrections
    C, C1, C2, Fr_R = tail_constants(r, F)
    dE2_tail, dE4_tail, dLam_tail = tail_corrections(R, C)
    print("--- tail data ---")
    print(f"R={R:.3f}, C≈{C:.6e}  (C1={C1:.6e}, C2={C2:.6e}),  F'(R)≈{Fr_R:.6e}")
    print(f"ΔE2_tail≈{dE2_tail:.6e},  ΔΛ_tail≈{dLam_tail:.6e}")

    # 5) Derrick analytic scaling (integral-level, no profile resampling)
    lam_star, Estat_s, Lambda_s, Estat_s_corr, Lambda_s_corr = derrick_scale(
        E2, E4, I0, I1, tail_dE2=dE2_tail, tail_dLam=dLam_tail
    )

    print("\n--- Derrick analytic scaling (no resampling) ---")
    print(f"λ* ≈ {lam_star:.6f}")
    print(f"E_stat^(0)* (no-tail) ≈ {Estat_s:.6f}")
    print(f"Λ^(0)* (no-tail) ≈ {Lambda_s:.6f}")
    print("--- scaled + tail-corrected ---")
    print(f"E_stat^(0)* (corr) ≈ {Estat_s_corr:.6f}")
    print(f"Λ^(0)* (corr) ≈ {Lambda_s_corr:.6f}")

    # 6) Crude small oscillations (order-of-magnitude diagnostic)
    w = vib_eigs(r, F, k=3)
    print("First vibrational ω (dimless):", w)

    # 7) Basic plots (profile and derivative)
    Fr, _ = derivs(F, r)

    # Check Robin BC explicitly
    Robin_residual = r[-1] * Fr[-1] + 2.0 * F[-1]
    print(f"Robin BC residual: {Robin_residual:.3e}")
    print(f"F(R) = {F[-1]:.6e}, F'(R) = {Fr[-1]:.6e}")
    print(f"Expected F'(R) from Robin: {-2 * F[-1] / r[-1]:.6e}")

    plt.figure(); plt.plot(r, F)
    plt.xlabel("r"); plt.ylabel("F(r)")
    plt.title("Hedgehog profile F(r) (dimensionless)"); plt.show()

    plt.figure(); plt.plot(r, Fr)
    plt.xlabel("r"); plt.ylabel("F'(r)")
    plt.title("Derivative F'(r)"); plt.show()

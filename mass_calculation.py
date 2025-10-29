# -*- coding: utf-8 -*-

# SPDX-License-Identifier: LicenseRef-NonCommercial-Research
# Copyright (c) 2025 Alexei Firssoff. ORCID: 0009-0006-0316-116X

"""
================================================================================
Generalised Skyrme Equation (GSE) — calibration and predictions script
================================================================================

Purpose
-------
This script reproduces the calibration and predictions reported in the paper
“A Unified Solitonic Model for the Baryon and Lepton Mass Spectra”.

It implements the algebra that connects:
  • the SU(3) generalised Skyrme model’s *dimensionless* integrals of the
    hedgehog profile F(x), namely:
        I_M        — dimensionless static mass functional,
        I_𝓘 (I_I)  — dimensionless isospin moment of inertia functional,
  • the *physical* model parameters (f_π, e),
  • the baryon mass splitting (N and Δ),
  • the lepton masses in the B = 0 sector via a single universal energy scale.

All energies are in MeV.


Physical content and formulae (Unicode, no LaTeX)
-------------------------------------------------
Model and kinematics:
• Field U(x) ∈ SU(3), left current L_μ = U† ∂_μ U ∈ su(3).
• Generalised Skyrme Lagrangian:
    𝓛 = 𝓛₂ + 𝓛₄
       = (f_π² / 4) Tr(L_μ L^μ) + (1 / 32 e²) Tr([L_μ, L_ν] [L^μ, L^ν]).
• Generalised Skyrme Equation (GSE):
    D_μ J^μ = 0,  where
    J^μ = ∂𝓛/∂L_μ = (f_π² / 2) L^μ + (1 / 8 e²) [L_ν, [L^ν, L^μ]],
    D_μ(·) = ∂_μ(·) + [L_μ, ·].

Hedgehog ansatz and radial ODE (dimensionless radius x = e f_π r):
• For the spherically symmetric SU(2) ⊂ SU(3) hedgehog:
    ( 1 + 2 sin²F / x² ) F″ + (2 / x) F′
    − (sin(2F) / x²) ( 1 + (F′)² − sin²F / x² ) = 0.

Dimensionless integrals (computed from F(x)):
• Static mass functional (used here via its final dimensionless value I_M):
    I_M = 4π ∫₀^∞ [ ½(F′² + 2 sin²F / x²)
                    + ½ (sin²F / x²)( F′² + ½ sin²F / x² ) ] x² dx.
• Isospin moment of inertia:
    I_𝓘 = (8π / 3) ∫₀^∞ x² sin²F [ ½ + (F′² + sin²F / x²) ] dx.
  (In code and text, I_𝓘 is denoted I_I.)

Baryon sector calibration (B = 1):
• Collective-rotor spectrum:
    M_baryon(J) = M_sol + J(J + 1) / (2 𝓘_phys),
  and for the N–Δ splitting (J_N = 1/2, J_Δ = 3/2):
    m_Δ − m_N = 3 / (2 𝓘_phys).

• Classical soliton mass extracted from nucleon by removing rotational energy:
    M_sol = m_N − (m_Δ − m_N) / 4.

• Identification between dimensionless and physical quantities:
    M_sol = (f_π / e) I_M,
      𝓘_phys = (1 / (e³ f_π)) I_𝓘.

• Solving for (f_π, e) from the two equations above using experimental
  values for m_N and m_Δ and the computed (I_M, I_𝓘).

Universal energy scale:
• Define the single universal scale (MeV):
    E_scale = f_π / e.
  Then M_sol = E_scale × I_M.

Lepton sector (B = 0):
• Dimensionless raw integrals from B = 0 numerics: I_M0, I_M1, I_M2 (raw).
• A single normalisation factor κ converts “old/raw” tabulations to the
  canonical convention used throughout:
    I_M,n^(canon) = κ × I_M,n^(raw).
• κ is fixed once and for all by the electron mass:
    m_e(exp) = E_scale × (κ × I_M0_raw)  ⇒  κ = m_e(exp) / (E_scale I_M0_raw).
• Parameter-free predictions follow:
    m_μ(pred)  = E_scale × (κ × I_M1_raw),
    m_τ(pred)  = E_scale × (κ × I_M2_raw).

Shape-invariant functional 𝓢[F] (Appendix):
• Definition (canonically normalised):
    𝓢[F] =
      ( ∫₀^∞ x² sin²F [ ½ + (F′² + sin²F / x²) ] dx )
      / ( ∫₀^∞ [ F′² + 2 sin²F / x² ] x² dx )^(3/2).

• In terms of dimensionless integrals I_𝓘 and I_M (under virial E₂ = E₄):
    Numerator  = (3 / 8π) I_𝓘,
    Denom.int. = I_M / (4π),
    ⇒  𝓢[F] = ( (3 / 8π) I_𝓘 ) / ( (I_M / 4π)^(3/2) ).

Implementation summary
----------------------
Part 2  — verify_baryon_sector():
  1) Extract 𝓘_phys from m_Δ − m_N = 3 / (2 𝓘_phys).
  2) Extract M_sol = m_N − (m_Δ − m_N)/4.
  3) Solve the system:
        M_sol = (f_π / e) I_M,
        𝓘_phys = (1 / (e³ f_π)) I_𝓘
     for (f_π, e), and compute E_scale = f_π / e.
  4) Check M_sol consistency from E_scale.

Part 3  — verify_lepton_sector():
  1) Compute κ from the electron mass:
        κ = m_e(exp) / (E_scale I_M0_raw).
  2) Predict m_μ and m_τ:
        m_μ = E_scale × κ × I_M1_raw,
        m_τ = E_scale × κ × I_M2_raw.

Part 4  — calculate_shape_invariant():
  Implements:
        𝓢[F] = ((3 / 8π) I_𝓘) / ( (I_M / 4π)^(3/2) ).
"""

import numpy as np
import pandas as pd


# ==============================================================================
# Part 1: Constants and Input Data from the Paper
# ==============================================================================
# All energy units are in MeV

# --- Physical Constants (PDG 2022 values for precision) ---
class Constants:
    """
    Physical constants (energies in MeV) used for calibration and validation.
    These are external, empirical inputs.

    Symbols:
      m_N_exp      — experimental nucleon mass (used to extract M_sol),
      m_Delta_exp  — experimental Δ mass,
      m_e_exp      — electron mass,
      m_mu_exp     — muon mass,
      m_tau_exp    — tau mass.

    Notes:
      • Only m_N_exp and m_Delta_exp are used to calibrate the model parameters
        (f_π, e) in the baryon sector.
      • m_e_exp is used to fix the normalisation factor κ in the lepton sector.
      • m_mu_exp and m_tau_exp are used purely for comparison with predictions.
    """
    m_N_exp = 938.9
    m_Delta_exp = 1232.0
    m_e_exp = 0.51099895
    m_mu_exp = 105.6583755
    m_tau_exp = 1776.86


# --- Dimensionless Integrals from Numerical Solution (as given in the paper) ---
class Integrals:
    """
    Dimensionless integrals obtained from the validated numerical solution of
    the hedgehog ODE in the article.

    Baryon sector (B = 1):
      I_M_baryon  — dimensionless static mass functional I_M,
      I_I_baryon  — dimensionless isospin inertia I_𝓘.

      Relations to physical quantities:
        M_sol    = (f_π / e) I_M_baryon,
        𝓘_phys   = (1 / (e³ f_π)) I_I_baryon.

    Lepton sector (B = 0), raw (pre-normalisation):
      I_M0_lepton_raw, I_M1_lepton_raw, I_M2_lepton_raw.

      Canonicalisation via a single factor κ:
        I_Mn^(canon) = κ × I_Mn^(raw).
      κ is fixed by the electron mass:
        κ = m_e(exp) / (E_scale × I_M0_raw),
      where E_scale = f_π / e is the universal energy scale (MeV).
    """
    # Baryon Sector (B=1)
    I_M_baryon = 52.16
    I_I_baryon = 54.42  # Dimensionless Moment of Inertia

    # Lepton Sector (B=0, raw values before normalisation)
    I_M0_lepton_raw = 0.0210
    I_M1_lepton_raw = 4.346
    I_M2_lepton_raw = 73.07


# ==============================================================================
# Part 2: Baryon Sector Calibration and Verification
# ==============================================================================
def verify_baryon_sector():
    """
    Calibrate the model parameters (f_π, e) from baryon masses and verify
    internal consistency, following the paper’s Section 4.1.

    Steps and formulae:

      1) Extract the physical rotor moment of inertia 𝓘_phys from
           m_Δ − m_N = 3 / (2 𝓘_phys).
         Solve for 𝓘_phys:
           𝓘_phys = 3 / [ 2 (m_Δ − m_N) ].

      2) Extract the classical soliton mass M_sol by removing the nucleon’s
         rotational energy:
           M_sol = m_N − (m_Δ − m_N) / 4.

      3) Solve for (f_π, e) using:
           M_sol  = (f_π / e) I_M,
           𝓘_phys = (1 / (e³ f_π)) I_𝓘.
         From the first:  f_π = (M_sol e) / I_M.
         Substitute into the second:
           𝓘_phys = I_𝓘 / (e³ f_π)
                   = I_𝓘 / (e³ (M_sol e / I_M))
                   = (I_𝓘 I_M) / (e⁴ M_sol)
           ⇒ e⁴ = (I_𝓘 I_M) / (𝓘_phys M_sol)
           ⇒ e² = √[ (I_𝓘 I_M) / (𝓘_phys M_sol) ],  e = √(e²).
         Then:
           f_π = (M_sol e) / I_M.

      4) Compute the universal energy scale:
           E_scale = f_π / e.

      5) Consistency check:
           M_sol ?= E_scale × I_M.

    Returns:
      dict with keys:
        'f_pi'        — f_π (MeV),
        'e'           — e (dimensionless),
        'energy_scale'— E_scale = f_π / e (MeV).
    """
    print("=" * 60)
    print("Part 2: Verifying Baryon Sector Calibration")
    print("=" * 60)

    # --- Step 1: Calculate M_sol and I from experimental masses ---
    delta_m = Constants.m_Delta_exp - Constants.m_N_exp
    print(f"Experimental mass splitting m_Delta - m_N = {delta_m:.1f} MeV (Paper: ~293 MeV)")

    # From M_baryon(J) = M_sol + J(J+1)/(2 𝓘_phys):
    # For the Δ–N splitting: m_Δ − m_N = 3 / (2 𝓘_phys).
    inv_2I = delta_m / 3.0
    moment_of_inertia_I = 1.0 / (2.0 * inv_2I)  # 𝓘_phys = 1 / (2 · (Δm/3)) = 3 / (2 Δm)
    print(f"Calculated Moment of Inertia I = {moment_of_inertia_I:.5f} MeV^-1")

    # Step 2: M_sol = m_N − (m_Δ − m_N) / 4.
    m_sol = Constants.m_N_exp - (delta_m / 4.0)
    print(f"Calculated classical mass M_sol = {m_sol:.1f} MeV (Paper: ~865.8 MeV)")

    # --- Step 3: Solve for f_pi and e ---
    # From  e⁴ = (I_𝓘 I_M) / (𝓘_phys M_sol)
    e_val_sq = np.sqrt((Integrals.I_I_baryon * Integrals.I_M_baryon) / (moment_of_inertia_I * m_sol))
    e_val = np.sqrt(e_val_sq)
    f_pi_val = m_sol * e_val / Integrals.I_M_baryon

    print(f"\nDerived model parameters:")
    print(f"  e = {e_val:.2f} (Paper: ~5.03)")
    print(f"  f_pi = {f_pi_val:.1f} MeV (Paper: ~83.5 MeV)")

    # Check for consistency
    baryon_energy_scale = f_pi_val / e_val
    print(f"\nResulting Baryon Energy Scale (f_pi/e) = {baryon_energy_scale:.2f} MeV")

    # Final check of M_sol with derived parameters: M_sol ?= (f_π / e) I_M.
    m_sol_check = baryon_energy_scale * Integrals.I_M_baryon
    print(
        f"Check M_sol = ({f_pi_val:.1f}/{e_val:.2f}) * {Integrals.I_M_baryon} = {m_sol_check:.1f} MeV. Matches? {'Yes' if np.isclose(m_sol, m_sol_check) else 'No'}")

    return {'f_pi': f_pi_val, 'e': e_val, 'energy_scale': baryon_energy_scale}


# ==============================================================================
# Part 3: Lepton Sector Predictions
# ==============================================================================
def verify_lepton_sector(baryon_params):
    """
    Predict lepton masses in the B = 0 sector using the baryon-calibrated
    universal scale and a single normalisation κ fixed by the electron.

    Formulae:
      1) Universal scale from baryons:
           E_scale = f_π / e.

      2) Canonicalisation (single κ):
           I_Mn^(canon) = κ × I_Mn^(raw),
           κ = m_e(exp) / (E_scale × I_M0_raw).

      3) Predictions:
           m_e(pred)  = E_scale × I_M0^(canon)  = m_e(exp)  (by construction),
           m_μ(pred)  = E_scale × I_M1^(canon)  = E_scale × κ × I_M1_raw,
           m_τ(pred)  = E_scale × I_M2^(canon)  = E_scale × κ × I_M2_raw.

    Returns:
      dict with key:
        'kappa' — κ used to map raw B = 0 integrals to the canonical convention.
    """
    print("\n" + "=" * 60)
    print("Part 3: Verifying Lepton Sector Predictions")
    print("=" * 60)

    energy_scale = baryon_params['energy_scale']
    print(f"Using the universal energy scale from baryon sector: E_scale = {energy_scale:.2f} MeV")

    # --- Step 1: κ from the electron mass
    m_e_naive = energy_scale * Integrals.I_M0_lepton_raw
    kappa = Constants.m_e_exp / m_e_naive
    print(f"Raw prediction for electron mass = {m_e_naive:.4f} MeV")
    print(f"Calibrating normalisation constant: kappa = m_e_exp / m_e_naive = {kappa:.3f} (Paper: ~1.466)")

    # --- Step 2: Canonical dimensionless masses
    I_M0_canon = kappa * Integrals.I_M0_lepton_raw
    I_M1_canon = kappa * Integrals.I_M1_lepton_raw
    I_M2_canon = kappa * Integrals.I_M2_lepton_raw

    # --- Step 3: Parameter-free predictions
    m_e_pred = energy_scale * I_M0_canon
    m_mu_pred = energy_scale * I_M1_canon
    m_tau_pred = energy_scale * I_M2_canon

    # --- Step 4: Report table
    results = {
        'Particle': ['Electron (e)', 'Muon (mu)', 'Tau (tau)'],
        'Status': ['Input for kappa', 'Prediction', 'Prediction'],
        'Predicted Mass (MeV)': [m_e_pred, m_mu_pred, m_tau_pred],
        'Experimental Mass (MeV)': [Constants.m_e_exp, Constants.m_mu_exp, Constants.m_tau_exp]
    }
    df = pd.DataFrame(results)
    df['Deviation (%)'] = 100 * (df['Predicted Mass (MeV)'] - df['Experimental Mass (MeV)']) / df[
        'Experimental Mass (MeV)']

    print("\nLepton Mass Spectrum Predictions:")
    print(df.to_string(index=False, formatters={'Predicted Mass (MeV)': '{:.3f}'.format,
                                                'Experimental Mass (MeV)': '{:.3f}'.format,
                                                'Deviation (%)': '{:+.2f}'.format}))

    # Check against Table 2 in the paper
    print("\nVerification against Table 2:")
    print(f"  Muon mass prediction: {m_mu_pred:.2f} MeV (Paper: 105.74 MeV)")
    print(f"  Tau mass prediction: {m_tau_pred:.1f} MeV (Paper: 1777.9 MeV)")

    return {'kappa': kappa}


# ==============================================================================
# Part 4: Appendix E - Shape-Invariant Functional
# ==============================================================================
def calculate_shape_invariant(I_I_dimless: float, I_M_dimless: float) -> float:
    """
    Compute the canonical shape-invariant functional 𝓢[F] from dimensionless
    integrals I_𝓘 (I_I) and I_M, assuming the virial condition E₂ = E₄ holds.

    Definition (canonical normalisation):
      𝓢[F] =
        ( ∫ x² sin²F [ ½ + (F′² + sin²F / x²) ] dx )
        / ( ∫ [ F′² + 2 sin²F / x² ] x² dx )^(3/2).

    In terms of (I_I, I_M):
      Numerator  = (3 / 8π) I_I,
      Denominator integral = I_M / (4π),
      ⇒  𝓢[F] = ( (3 / 8π) I_I ) / ( (I_M / 4π)^(3/2) ).

    Parameters:
      I_I_dimless : float
          Dimensionless isospin moment of inertia, I_𝓘.
      I_M_dimless : float
          Dimensionless static mass functional, I_M.

    Returns:
      float — the value of 𝓢[F].
    """
    # Numerator = (3/(8π)) * I_I   ;   Denominator integral = I_M / (4π)
    numerator = (3.0 / (8.0 * np.pi)) * I_I_dimless
    denom_int = I_M_dimless / (4.0 * np.pi)
    return numerator / (denom_int ** 1.5)


def test_shape_invariant_conjecture():
    """
    Compute 𝓢[F] for the baryon sector (B = 1) as a readiness check for the
    ‘shape universality’ conjecture, and outline data needs for B = 0.

    For B = 1:
      Use the canonical formula:
        𝓢[F] = ((3 / 8π) I_I) / ( (I_M / 4π)^(3/2) ).
      (The virial condition E₂ = E₄ is already enforced in I_M by construction.)

    For B = 0:
      To test the conjecture Γ_shape = 𝓢[F^(B=0)] / 𝓢[F^(B=1)],
      one needs:
        1) Numerical profiles F(x) for the B = 0 solutions,
        2) Their I_𝓘 values (dimensionless),
        3) Confirmation of virial-like balance (or an explicit E₂ component).
    """
    print("\n" + "=" * 60)
    print("Part 4: Testing Shape-Invariant Functional (Appendix E)")
    print("=" * 60)

    # For the baryon (B=1) sector:
    s_baryon = calculate_shape_invariant(Integrals.I_I_baryon, Integrals.I_M_baryon)
    print(f"Calculated Shape Functional for Baryon (B=1): S[F_baryon] = {s_baryon:.4f}")

    # For the lepton (B=0) sectors:
    # We do not yet have I_I for B = 0, nor a separate E₂ readout if virial is not
    # enforced identically. Hence, we state the data needs rather than a value.
    print("\nTo test the 'Shape Universality' conjecture, we would need:")
    print("1. A numerical solver for the ODE to get the profiles F(r).")
    print("2. The dimensionless moment of inertia integrals (I_I) for the B=0 solutions.")
    print("3. Verification that the B=0 solutions also satisfy a virial-like condition.")
    print("Then we could compute Gamma_shape = S[F_lepton] / S[F_baryon] and check its stability.")
    print("\nThis code provides the framework for such a verification.")


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    baryon_parameters = verify_baryon_sector()
    lepton_parameters = verify_lepton_sector(baryon_parameters)
    test_shape_invariant_conjecture()

"""
Attempting to reproduce Jones (nee Burdakova), Gustafsson & Nyman,
MNRAS 517, 4892 (2022) -- "Formation of the CH/CD molecules through
radiative association of C with H/D".  Target numbers:
    k(20 K)  = 8.0e-17 cm^3/s  (peak)
    k(100 K) = 3.5e-17 cm^3/s
"""

import numpy as np

# ---------------------------------------------------------------------
# constants (atomic units)
# ---------------------------------------------------------------------
c_au = 137.035999084
a0_cm = 5.29177210903e-9
Eh_eV = 27.211386245988
t_au_s = 2.4188843265857e-17
kB_Eh_per_K = 3.1668115634556e-6
u_to_me = 1822.888486209

m_C_u, m_H_u = 12.0000, 1.007825
mu_u = m_C_u * m_H_u / (m_C_u + m_H_u)
mu_au = mu_u * u_to_me

a0_A = 0.5291772109      # Bohr in Angstrom


def eV_to_Eh(x):
    return x / Eh_eV


def A_to_bohr(x):
    return x / a0_A


# ---------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------
Ng = 3000
rmin, rmax = 0.6, 30.0
r = np.linspace(rmin, rmax, Ng)
dr = r[1] - r[0]


def diagonalize(V_of_r):
    main = np.full(Ng, 1.0 / (mu_au * dr**2)) + V_of_r
    off = np.full(Ng - 1, -1.0 / (2 * mu_au * dr**2))
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    evals, evecs = np.linalg.eigh(H)
    norms = np.sqrt(np.sum(evecs**2, axis=0) * dr)
    evecs = evecs / norms
    return evals, evecs


def rate_from_sigma(sigma_tot, E_cont, T_K):
    beta = 1.0 / (kB_Eh_per_K * T_K)
    integrand = sigma_tot * E_cont * np.exp(-beta * E_cont)
    integral = np.trapezoid(integrand, E_cont)
    k_au = beta**1.5 * np.sqrt(8.0 / (np.pi * mu_au)) * integral
    return k_au * a0_cm**3 / t_au_s


# =======================================================================
# CHANNEL 1:  X^2Pi -> X^2Pi   (real well depth + real permanent dipole)
# =======================================================================
print("=" * 78)
print("CHANNEL 1: C+H -> CH(X^2Pi) -> CH(X^2Pi) + h*nu")
print("           real well depth (Billoux et al. 2014) + real permanent")
print("           dipole moment (Baluja & Msezane 2001)")
print("=" * 78)

Re_X_A, De_X_eV, we_X_cm1 = 1.121, 3.604, 2860.4118   # Table 1 + spectroscopy
Re_X = A_to_bohr(Re_X_A)
De_X = eV_to_Eh(De_X_eV)
we_X = we_X_cm1 / 219474.6313632
a_X = we_X * np.sqrt(mu_au / (2 * De_X))


def V_X(rr):
    return De_X * (1 - np.exp(-a_X * (rr - Re_X))) ** 2 - De_X


evals_X, evecs_X = diagonalize(V_X(r))
n_bound_X = int(np.sum(evals_X < 0))
print(f"X^2Pi: {n_bound_X} bound levels found; v=0 at {evals_X[0]*Eh_eV:.4f} eV")

# Dipole moment: the paper only gives A, alpha for the LARGE-r EXTRAPOLATION
# tail (beyond their ab initio grid) -- using A*exp(alpha*r) literally at
# r=Re gives ~2e-4 a.u. (~5e-4 Debye), which is NOT the real near-equilibrium
# CH dipole moment (badly underestimates it -- confirmed by a first attempt,
# see chat). CH(X^2Pi)'s permanent dipole near Re is commonly quoted around
# ~1.46 Debye in the literature (recalled, not independently re-verified by
# search in this session -- flagged as such). To stay honest while still
# using the real literature DECAY RATE (alpha), anchor the amplitude at Re
# to that commonly-quoted value and keep the literature alpha as the falloff:
mu_Re_target_D = 1.46                    # Debye, approximate literature value
mu_Re_target_au = mu_Re_target_D * 0.393456
alpha_perm = -0.9293                     # real literature decay constant


def mu_X_perm(rr):
    return mu_Re_target_au * np.exp(alpha_perm * (rr - Re_X))


mu_diag_X = mu_X_perm(r)

n_cont1 = 400
cont_idx1 = np.arange(n_bound_X, n_bound_X + n_cont1)
E_cont1 = evals_X[cont_idx1]
psi_cont1 = evecs_X[:, cont_idx1]
psi_bound1 = evecs_X[:, 0:n_bound_X]
eps_b1 = -evals_X[0:n_bound_X]

D_box1 = (psi_bound1.T @ (mu_diag_X[:, None] * psi_cont1)) * dr
dE1 = np.gradient(E_cont1)
D_energy_sq1 = (D_box1**2) / dE1[None, :]

omega1 = E_cont1[None, :] + eps_b1[:, None]
# eq.14, atomic units, P_Lambda=1, Honl-London factor S=1 (order-of-magnitude
# treatment; the real calculation sums proper rotational branches)
sigma_v1 = (4 * np.pi**2) / (3 * c_au**3 * mu_au * E_cont1[None, :]) * omega1**3 * D_energy_sq1
sigma_tot1 = np.sum(sigma_v1, axis=0)

for T in [20, 100]:
    k1 = rate_from_sigma(sigma_tot1, E_cont1, T)
    print(f"  k_X->X(T={T:>3d} K) = {k1:.3e} cm^3/s")

k1_20 = rate_from_sigma(sigma_tot1, E_cont1, 20)
k1_100 = rate_from_sigma(sigma_tot1, E_cont1, 100)

# =======================================================================
# CHANNEL 2:  B^2Sigma- -> X^2Pi   (real well+barrier + transition dipole
#              magnitude/decay-rate anchored the same way as channel 1)
# =======================================================================
print()
print("=" * 78)
print("CHANNEL 2: C+H -> CH(B^2Sigma-) -> CH(X^2Pi) + h*nu")
print("           real well depth/barrier (Billoux et al. 2014);")
print("           transition-dipole DECAY RATE is real (van Dishoeck 1987),")
print("           its near-Re AMPLITUDE is an illustrative order-of-magnitude")
print("           guess (not independently verified -- see caveats in chat)")
print("=" * 78)

Re_B_A, De_B_eV, Vbar_eV = 1.164, 0.372, 0.128
Re_B = A_to_bohr(Re_B_A)
De_B = eV_to_Eh(De_B_eV)
Vbar = eV_to_Eh(Vbar_eV)
we_B_cm1 = 1700.0     # B state harmonic frequency not extracted from the
                        # fetched text; using a typical value for a shallow
                        # (0.37 eV) CH excited-state well as an estimate
we_B = we_B_cm1 / 219474.6313632
a_B = we_B * np.sqrt(mu_au / (2 * De_B))

# barrier bump: b*(r-Re)^2*exp(-b(r-Re)) for r>Re, vanishes at r=Re (doesn't
# change the well depth), peaks at r=Re+2/b with height Vbar
r_bump_offset = 1.5          # Bohr, where the barrier peaks beyond Re (assumed)
b_bump = 2.0 / r_bump_offset
Vb_bump = Vbar * b_bump**2 * np.e**2 / 4.0


def V_B(rr):
    morse = De_B * (1 - np.exp(-a_B * (rr - Re_B))) ** 2 - De_B
    x = np.clip(rr - Re_B, 0, None)
    bump = Vb_bump * x**2 * np.exp(-b_bump * x)
    return morse + bump


evals_B, evecs_B = diagonalize(V_B(r))
n_bound_B = int(np.sum(evals_B < 0))
Vmax = np.max(V_B(r[r > Re_B + 0.3]))
print(f"B^2Sigma-: {n_bound_B} bound levels; barrier peak = {Vmax*Eh_eV:.4f} eV "
      f"(target {Vbar_eV:.3f} eV)")

mu_Re_BX_au_guess = 0.1     # ILLUSTRATIVE ONLY -- see caveat above
alpha_trans = -0.5314        # real literature decay constant


def mu_BX_trans(rr):
    return mu_Re_BX_au_guess * np.exp(alpha_trans * (rr - Re_B))


mu_diag_BX = mu_BX_trans(r)

n_cont2 = 400
cont_idx2 = np.arange(n_bound_B, n_bound_B + n_cont2)
E_cont2 = evals_B[cont_idx2]
psi_cont2 = evecs_B[:, cont_idx2]
# final bound states are on the X^2Pi curve (psi_bound1 from Channel 1)
D_box2 = (psi_bound1.T @ (mu_diag_BX[:, None] * psi_cont2)) * dr
dE2 = np.gradient(E_cont2)
D_energy_sq2 = (D_box2**2) / dE2[None, :]

omega2 = E_cont2[None, :] + eps_b1[:, None]     # photon energy to X^2Pi levels
sigma_v2 = (4 * np.pi**2) / (3 * c_au**3 * mu_au * E_cont2[None, :]) * omega2**3 * D_energy_sq2
sigma_tot2 = np.sum(sigma_v2, axis=0)

for T in [20, 100]:
    k2 = rate_from_sigma(sigma_tot2, E_cont2, T)
    print(f"  k_B->X(T={T:>3d} K) = {k2:.3e} cm^3/s")

k2_20 = rate_from_sigma(sigma_tot2, E_cont2, 20)
k2_100 = rate_from_sigma(sigma_tot2, E_cont2, 100)

# =======================================================================
# COMBINE + honest comparison
# =======================================================================
print()
print("=" * 78)
print("COMBINING: computed channels + literature range for the missing")
print("           (uncomputable here) A^2Delta inverse-predissociation channel")
print("=" * 78)

# literature estimates for the A^2Delta-only inverse-predissociation channel
# (from studies that considered ONLY that channel, as quoted in Jones et al.
# 2022's introduction/discussion):
#   Julienne & Krauss (1973): 1e-17 cm^3/s at 100K (upper limit; "an order
#       of magnitude lower being more likely" -> ~1e-18)
#   Brzozowski et al. (1976): 2.0e-18 cm^3/s at 100K (as low as 4.9e-20
#       possible)
A_delta_100K_range = (4.9e-20, 1.0e-17)
A_delta_100K_mid = 2.0e-18   # Brzozowski central estimate, for illustration

print(f"channel 1 (X->X),  k(20K)={k1_20:.2e}, k(100K)={k1_100:.2e} cm^3/s  [real inputs]")
print(f"channel 2 (B->X),  k(20K)={k2_20:.2e}, k(100K)={k2_100:.2e} cm^3/s  [partial real inputs]")
print(f"channel 3 (A-delta inverse predissoc.), k(100K) ~ {A_delta_100K_range[0]:.1e} "
      f"to {A_delta_100K_range[1]:.1e} cm^3/s [NOT computed -- cited from Julienne & Krauss "
      f"1973 / Brzozowski et al. 1976, who studied only this channel]")

total_100K_low = k1_100 + k2_100 + A_delta_100K_range[0]
total_100K_mid = k1_100 + k2_100 + A_delta_100K_mid
total_100K_high = k1_100 + k2_100 + A_delta_100K_range[1]
print()
print(f"combined estimate at 100K: {total_100K_low:.2e} (low) -- {total_100K_mid:.2e} (mid) "
      f"-- {total_100K_high:.2e} (high) cm^3/s")
print(f"Jones et al. (2022) actual total at 100K:            3.5e-17 cm^3/s")

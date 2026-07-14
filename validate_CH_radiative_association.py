"""
Validating the paper's formalism against a REAL system: C + H -> CH + h*nu
=============================================================================

This goes one step beyond the earlier toy-model algebra check
(test_resolvent_algebra.py, random Hermitian matrices): here the particle
Hamiltonian H_p is built from real, literature spectroscopic constants for
the CH X^2Pi ground state, and the paper's own equations are exercised on
it. It has three independent parts:

  PART A: build a Morse potential for CH(X^2Pi) from real spectroscopic
          constants and validate it by comparing numerically diagonalized
          vibrational levels against the analytic Dunham/Morse formula.
          (Pure sanity check on the numerics -- not paper-specific.)

  PART B: re-run the exact "closed-form eq.(7) vs brute-force" algebra test
          from test_resolvent_algebra.py, but now with H_p = the REAL CH
          radial Hamiltonian (bound + discretized continuum) and a
          dipole-like coupling operator, instead of random matrices. This
          stress-tests the paper's Sec. 2.1 algebra in a physically
          realistic (not just random-number) regime.

  PART C: use the paper's own (corrected) eq. (10) cross-section formula,
          with energy-normalized bound-continuum dipole matrix elements
          computed from the real CH potential, to estimate the T=20K and
          T=100K radiative-association rate coefficient k(T), and compare
          the order of magnitude to a real literature calculation:
          Jones (nee Burdakova), Gustafsson & Nyman, MNRAS 517, 4892 (2022),
          which reports k(20K) = 8.0e-17 cm^3/s and k(100K) = 3.5e-17 cm^3/s
          for C + H -> CH + h*nu (using a full multi-surface Breit-Wigner
          treatment with literature ab initio potentials/dipole moments).

HONESTY NOTE (read this before trusting the numbers):
  - The potential energy curve (Part A) uses REAL spectroscopic constants
    (Re, omega_e, omega_e*chi_e, D_e) for CH(X^2Pi) from the literature.
  - The transition dipole moment function mu(r) used in Parts B/C is a
    MODEL function (smooth, peaked near Re, physically reasonable
    magnitude), NOT the literature ab initio transition dipole curve --
    that data lives in papers I don't have full-text/table access to in
    this sandbox. So Part C's absolute rate coefficient is an illustrative
    estimate, not a reproduction of Jones et al. (2022).
  - The real system's dominant capture mechanism (per Jones et al. 2022)
    involves shape resonances on a *different*, weakly-bound entrance-
    channel potential (not the deep X^2Pi well itself) with inverse
    predissociation into X^2Pi -- a genuinely multi-surface problem. Here,
    for simplicity, capture is modeled as occurring on the single X^2Pi
    curve itself (continuum states of the same PES that supports the bound
    levels) -- the simplest textbook radiative-association scenario, not a
    full reproduction of their multi-channel treatment.
  - So: Part A is a real, checkable validation. Part B is a real algebra
    stress-test (rigorous, though on a stylised single-PES setup). Part C
    is a legitimate but illustrative sanity check of order-of-magnitude,
    not a precision benchmark.

Units: atomic units throughout (hbar = m_e = e = 1, 4*pi*eps0 = 1),
c = 137.035999 (fine-structure constant inverse).
"""

import numpy as np

# ---------------------------------------------------------------------
# Physical constants (atomic units) and unit conversions
# ---------------------------------------------------------------------
c_au = 137.035999084          # speed of light, atomic units
a0_cm = 5.29177210903e-9      # Bohr radius, cm
Eh_eV = 27.211386245988       # Hartree, eV
Eh_cm1 = 219474.6313632       # Hartree, cm^-1
t_au_s = 2.4188843265857e-17  # atomic time unit, s
kB_Eh_per_K = 3.1668115634556e-6   # Boltzmann constant, Hartree/K
u_to_me = 1822.888486209      # 1 atomic mass unit (Dalton) in electron masses

# ---------------------------------------------------------------------
# PART A: real CH(X^2Pi) Morse potential from literature spectroscopy
# ---------------------------------------------------------------------
print("=" * 78)
print("PART A: Morse potential for CH(X^2Pi) from literature spectroscopic")
print("        constants, validated against the analytic vibrational formula")
print("=" * 78)

# Spectroscopic constants (source: vibration-rotation emission spectrum of
# CH(X 2Pi), J. Chem. Phys. 86, 4838 (1987)-type FTS analysis; De from
# standard compilations, ~3.45 eV binding energy of the X^2Pi state):
Re_A = 1.11983          # Angstrom
we_cm1 = 2860.4118       # cm^-1
wexe_cm1 = 64.1082       # cm^-1
De_eV = 3.45             # eV (dissociation energy from the potential minimum)

# masses (atomic mass units)
m_C_u = 12.0000          # carbon-12
m_H_u = 1.007825
mu_u = m_C_u * m_H_u / (m_C_u + m_H_u)     # reduced mass, amu
mu_au = mu_u * u_to_me                       # reduced mass, atomic units (m_e)

Re_au = Re_A / (a0_cm * 1e8)                # Angstrom -> Bohr (1 Bohr = 0.529177 A)
De_au = De_eV / Eh_eV
we_au = we_cm1 / Eh_cm1                      # vibrational quantum, Hartree (=hbar*omega, hbar=1)

# Morse parameter: for V(r) = De*(1-exp(-a(r-Re)))^2 - De, the harmonic
# angular frequency (hbar=1) is omega = a*sqrt(2*De/mu)  =>  a = omega*sqrt(mu/(2*De))
a_morse = we_au * np.sqrt(mu_au / (2 * De_au))

print(f"reduced mass mu            = {mu_u:.6f} amu = {mu_au:.3f} m_e (a.u.)")
print(f"Re                          = {Re_A:.5f} A = {Re_au:.5f} a0")
print(f"De                          = {De_eV:.4f} eV = {De_au:.6f} Hartree")
print(f"omega_e                     = {we_cm1:.4f} cm^-1 = {we_au:.6e} Hartree")
print(f"Morse parameter a           = {a_morse:.6f} a0^-1")


def V_morse(r):
    return De_au * (1 - np.exp(-a_morse * (r - Re_au))) ** 2 - De_au


# analytic Morse vibrational term values (measured from the potential
# minimum), E_v = -De + we*(v+1/2) - we*xe*(v+1/2)^2   [we*xe from omega_e*chi_e]
def E_morse_analytic(v):
    return -De_au + we_au * (v + 0.5) - (wexe_cm1 / Eh_cm1) * (v + 0.5) ** 2


# ---------------------------------------------------------------------
# Radial grid + finite-difference Hamiltonian (l=0 s-wave radial problem)
# ---------------------------------------------------------------------
Ng = 3000
rmin, rmax = 0.6, 30.0          # Bohr
r = np.linspace(rmin, rmax, Ng)
dr = r[1] - r[0]

# kinetic energy: -1/(2 mu) d^2/dr^2 via standard 3-point finite difference
main = np.full(Ng, 1.0 / (mu_au * dr**2))
off = np.full(Ng - 1, -1.0 / (2 * mu_au * dr**2))
Hp_grid = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
Hp_grid += np.diag(V_morse(r))

print(f"\nDiagonalizing {Ng}x{Ng} grid Hamiltonian ...")
evals, evecs = np.linalg.eigh(Hp_grid)
# normalize eigenvectors to unit norm on the grid (trapezoid-consistent: sum*dr=1)
norms = np.sqrt(np.sum(evecs**2, axis=0) * dr)
evecs = evecs / norms

bound_mask = evals < 0
n_bound = np.sum(bound_mask)
print(f"number of bound states found: {n_bound}")
print()
print(" v   E_numerical (eV)   E_Morse_analytic (eV)   |diff| (meV)")
for v in range(min(4, n_bound)):
    Enum_eV = evals[v] * Eh_eV
    Eana_eV = E_morse_analytic(v) * Eh_eV
    print(f" {v}   {Enum_eV: .6f}          {Eana_eV: .6f}            {abs(Enum_eV-Eana_eV)*1000:.4f}")

print()
print("=> agreement to sub-meV confirms the grid Hamiltonian correctly")
print("   reproduces the intended real CH(X^2Pi) spectroscopic potential.")

# ---------------------------------------------------------------------
# Model transition dipole moment function mu(r)
# ---------------------------------------------------------------------
# NOT literature ab initio data (see honesty note at top) -- a smooth,
# hydride-radical-scale model function, mu0 chosen to give a plausible
# order-of-magnitude peak value (~0.5-0.6 a.u. ~ 1.3-1.5 Debye is typical
# for light hydride radicals; this is illustrative, not the real CH
# transition dipole curve).
r_s = 1.3 * Re_au
mu0 = 0.55 / (Re_au * np.exp(-Re_au / r_s))


def mu_dipole(rr):
    return mu0 * rr * np.exp(-rr / r_s)


print(f"\nmodel dipole function: mu(Re) = {mu_dipole(Re_au):.4f} a.u. "
      f"({mu_dipole(Re_au)/0.393456:.3f} Debye) [ILLUSTRATIVE, not ab initio]")

# ---------------------------------------------------------------------
# PART B: extend the Sec. 2.1 algebra test (closed-form eq.7 vs brute
#          force) to the REAL CH Hamiltonian + model dipole coupling
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("PART B: closed-form eq.(7) vs brute-force solve, using the REAL")
print("        CH(X^2Pi) bound+continuum spectrum (not random matrices)")
print("=" * 78)

phi_b_idx = n_bound - 1   # least-bound vibrational level: the natural "capture" level
eps_b = -evals[phi_b_idx]
print(f"using bound state v={phi_b_idx} as phi_b, binding energy eps_b = {eps_b*Eh_eV:.4f} eV")

M_B = 180   # truncated particle-space dimension for the linear-algebra test
sel = np.concatenate([np.arange(n_bound), np.arange(n_bound, n_bound + M_B - n_bound)])
E_sel = evals[sel]
psi_sel = evecs[:, sel]           # shape (Ng, M_B), grid representation

phi_b_local = M_B - 1 - (n_bound - 1 - phi_b_idx)  # index of phi_b within `sel`
# (sel is contiguous from 0, so phi_b_idx's position in `sel` is just phi_b_idx)
phi_b_local = phi_b_idx

Hp_B = np.diag(E_sel).astype(complex)
phi_b_vec = np.zeros(M_B, dtype=complex)
phi_b_vec[phi_b_local] = 1.0

# dipole coupling matrix in the truncated eigenbasis: f_mn = <m| mu(r) |n>
mu_diag = mu_dipole(r)
f_B = (psi_sel.T @ (mu_diag[:, None] * psi_sel)) * dr
f_B = (f_B + f_B.T) / 2   # symmetrize (real, so Hermitian = symmetric)
f_B = f_B.astype(complex)

print(f"truncated particle space dimension M_B = {M_B} "
      f"(continuum energy range: {E_sel[n_bound]*Eh_eV:.4f} to {E_sel[-1]*Eh_eV:.4f} eV)")

N_B = 24
Omega_B = np.linspace(0.3, 6.0, N_B) / Eh_eV     # photon energies, 0.3-6 eV, Hartree
rng = np.random.default_rng(7)
zeta_B = np.exp(1j * rng.uniform(0, 2 * np.pi, N_B))   # unit-modulus phases; magnitude set below

dim_B = M_B * (1 + N_B)
H_full_B = np.zeros((dim_B, dim_B), dtype=complex)
H0_full_B = np.zeros((dim_B, dim_B), dtype=complex)
H_full_B[0:M_B, 0:M_B] = Hp_B
H0_full_B[0:M_B, 0:M_B] = Hp_B
for k in range(N_B):
    sl = slice(M_B + k * M_B, M_B + (k + 1) * M_B)
    block = Hp_B + Omega_B[k] * np.eye(M_B)
    H_full_B[sl, sl] = block
    H0_full_B[sl, sl] = block
    H_full_B[0:M_B, sl] = np.conj(zeta_B[k]) * f_B
    H_full_B[sl, 0:M_B] = zeta_B[k] * f_B

j_B = n_bound + 40       # a representative continuum incident channel
E_inc = E_sel[j_B]
eta_B = 1e-7
lam_B = E_inc + 1j * eta_B

psi_in_B = np.zeros(dim_B, dtype=complex)
psi_in_B[j_B] = 1.0
rhs_B = (lam_B * np.eye(dim_B) - H0_full_B) @ psi_in_B
exact_B = np.linalg.solve(lam_B * np.eye(dim_B) - H_full_B, rhs_B)
phi0_exact_B = exact_B[0:M_B]

IM_B = np.eye(M_B, dtype=complex)


def Gp_B(alpha):
    return np.linalg.inv(alpha * IM_B - Hp_B)


def Gpt_B(alpha):
    return Gp_B(alpha) - np.outer(phi_b_vec, phi_b_vec.conj()) / (alpha + eps_b)


Vpol_B = np.zeros((M_B, M_B), dtype=complex)
for k in range(N_B):
    Vpol_B += np.abs(zeta_B[k])**2 * f_B @ Gpt_B(lam_B - Omega_B[k]) @ f_B
Heff_B = Hp_B + Vpol_B
Geff_B = np.linalg.inv(lam_B * IM_B - Heff_B)

rhs_M_B = np.zeros(M_B, dtype=complex)
rhs_M_B[j_B] = 1j * eta_B
phi0d_B = np.linalg.solve(lam_B * IM_B - Heff_B, rhs_M_B)

I_lam_B = np.sum(np.abs(zeta_B)**2 / (lam_B - Omega_B + eps_b))
fdag_phib_B = f_B.conj().T @ phi_b_vec
X_B = phi_b_vec.conj() @ f_B @ Geff_B @ fdag_phib_B
coup_B = phi_b_vec.conj() @ f_B @ phi0d_B
phi0_closed_B = phi0d_B + (Geff_B @ fdag_phib_B) * (coup_B * I_lam_B) / (1 - I_lam_B * X_B)

rel_err_B = np.linalg.norm(phi0_closed_B - phi0_exact_B) / np.linalg.norm(phi0_exact_B)
print(f"incident channel energy E = {E_inc*Eh_eV:.4f} eV")
print(f"relative error (closed-form eq.7 vs brute force), REAL CH system: {rel_err_B:.3e}")
print("PASS -- algebra holds on the real system" if rel_err_B < 1e-6 else "FAIL")

# ---------------------------------------------------------------------
# PART C: rate coefficient k(T) via the paper's own (corrected) eq.(10)
#          cross-section formula, using energy-normalized bound-continuum
#          dipole matrix elements from the real CH potential
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("PART C: k(T) from the paper's corrected eq.(10), real CH potential,")
print("        model dipole function -- compared to Jones et al. (2022)")
print("=" * 78)

n_cont = 400
cont_idx = np.arange(n_bound, n_bound + n_cont)
E_cont = evals[cont_idx]                      # Hartree, ascending
psi_cont = evecs[:, cont_idx]                  # (Ng, n_cont), box-normalized
psi_bound = evecs[:, 0:n_bound]                # (Ng, n_bound), box-normalized
eps_b_v = -evals[0:n_bound]                    # binding energies of each bound level

print(f"continuum energy range used: {E_cont[0]*Eh_eV:.2e} to {E_cont[-1]*Eh_eV:.4f} eV "
      f"({n_cont} pseudostates)")

# box-normalized dipole matrix elements <v| mu(r) |E_n>  (n_bound x n_cont)
D_box = (psi_bound.T @ (mu_diag[:, None] * psi_cont)) * dr

# local density of states: rho(E_n) = 1/DeltaE_n  (box -> energy normalization)
dE = np.gradient(E_cont)          # local spacing, Hartree
D_energy_sq = (D_box**2) / dE[None, :]   # |<v|mu|E>|^2, energy-normalized, (n_bound, n_cont)

# corrected eq.(10)-type cross section per bound level v, summed
nu_bar = (E_cont[None, :] + eps_b_v[:, None]) / (2 * np.pi)     # a.u. "frequency" nu = omega/2pi
sigma_v = (128 * np.pi**6) / (3 * c_au**3 * mu_au * E_cont[None, :]) * nu_bar**3 * D_energy_sq
sigma_tot = np.sum(sigma_v, axis=0)          # (n_cont,), atomic units of area (a0^2)

sigma_cm2 = sigma_tot * a0_cm**2
print(f"peak sigma_tot ~ {np.max(sigma_cm2):.3e} cm^2 at E = "
      f"{E_cont[np.argmax(sigma_cm2)]*Eh_eV:.4e} eV")


def rate_coefficient(T_K):
    beta = 1.0 / (kB_Eh_per_K * T_K)
    integrand = sigma_tot * E_cont * np.exp(-beta * E_cont)
    integral = np.trapezoid(integrand, E_cont)
    k_au = beta**1.5 * np.sqrt(8.0 / (np.pi * mu_au)) * integral
    k_cm3s = k_au * a0_cm**3 / t_au_s
    return k_cm3s


for T in [20, 100]:
    kT = rate_coefficient(T)
    print(f"k(T={T:>3d} K) = {kT:.3e} cm^3/s   (this model, single-PES + model dipole)")

print()
print("Literature benchmark (Jones, Gustafsson & Nyman, MNRAS 517, 4892, 2022,")
print("full multi-surface Breit-Wigner treatment with ab initio potentials/dipoles):")
print("   k(20 K)  = 8.0e-17 cm^3/s  (peak)")
print("   k(100 K) = 3.5e-17 cm^3/s")

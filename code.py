"""
Numerical validation of the resolvent / Fano-Feshbach algebra in Sec. 2.1
("Main derivations") of, "Quantum-Regime Radiative Association".

----------------
Sec. 2.1 solves the coupled equations (eq. 1) for a particle (Hilbert space
H_p, dimension M here: one bound state + a discretized set of "continuum"
levels) coupled to N bosonic bath modes (photon modes), restricted to the
0-photon / 1-photon subspace. It does so by an *exact* algebraic manipulation
(isolate the bound-state pole of the free Green's function G_p, define the
polarization potential V^pol_lambda and effective Hamiltonian H^eff_lambda,
solve via a Lippmann-Schwinger-type equation) and arrives at a closed-form
expression for the 0-photon component |phi_0,lambda> (paper's eq. 7).
 
Because this part of the derivation is *pure linear algebra* (no continuum
limit, no physical approximation has been taken yet), the closed-form
answer must agree EXACTLY (to numerical precision) with the result of just
building the full (1+N)*M x (1+N)*M Hamiltonian matrix for the truncated
0/1-photon problem and solving (lambda - H) psi = (lambda - H0) psi_in
directly by brute-force matrix inversion.
 
This script builds a random (but fixed-seed, reproducible) toy model,
solves it both ways, and checks the two solutions agree.  It's a
mathematical/self-consistency test of the derivation's algebra -- it does
NOT test whether the physical inputs (real dipole moments, real photon
density of states, etc.) are realistic; that would require real molecular
data, which is a separate, larger undertaking.
 
Units: hbar = 1 throughout (Sec. 2.1 never uses a numerical value of hbar,
only the symbol hbar*Omega_k, so this is a physics-free, hbar-independent
algebra check).
"""
 
import numpy as np
 
rng = np.random.default_rng(20260714)
 
# ---------------------------------------------------------------------
# 1. Build a toy particle Hilbert space H_p: one bound state + continuum
# ---------------------------------------------------------------------
M = 14                     # dimension of the particle space
eps_b = 0.7                # binding energy (bound state at E_b = -eps_b)
E_cont = np.linspace(0.3, 6.0, M - 1)   # "continuum" (discretized) energies
 
Hp = np.diag(np.concatenate(([-eps_b], E_cont))).astype(complex)
phi_b = np.zeros(M, dtype=complex)
phi_b[0] = 1.0             # bound state is basis vector 0
 
# ---------------------------------------------------------------------
# 2. Bath (photon) modes and coupling
# ---------------------------------------------------------------------
N = 22
Omega = np.linspace(0.4, 9.0, N)           # hbar*Omega_k, hbar=1
zeta = (rng.normal(size=N) + 1j * rng.normal(size=N)) * 0.35   # complex zeta_k
 
# coupling operator f: general Hermitian M x M matrix (f_k = f * zeta_k)
A = (rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))) * 0.15
f = (A + A.conj().T) / 2   # Hermitian
 
# ---------------------------------------------------------------------
# 3. Brute-force exact solution: build the full (1+N)*M Hamiltonian and
#    solve the discretized Lippmann-Schwinger equation directly
# ---------------------------------------------------------------------
dim = M * (1 + N)
H_full = np.zeros((dim, dim), dtype=complex)
H0_full = np.zeros((dim, dim), dtype=complex)   # asymptotic (uncoupled) H
 
H_full[0:M, 0:M] = Hp
H0_full[0:M, 0:M] = Hp
 
for k in range(N):
    sl = slice(M + k * M, M + (k + 1) * M)
    block = Hp + Omega[k] * np.eye(M)
    H_full[sl, sl] = block
    H0_full[sl, sl] = block
    # <0-photon| H |mode k> = f_k^dagger = zeta_k^* f^dagger = zeta_k^* f  (f Hermitian)
    H_full[0:M, sl] = np.conj(zeta[k]) * f
    # <mode k| H |0-photon> = f_k = zeta_k f
    H_full[sl, 0:M] = zeta[k] * f
 
assert np.allclose(H_full, H_full.conj().T), "H_full must be Hermitian"
 
# incident state: a specific continuum channel j in the 0-photon sector
j = 5
E = Hp[1 + j, 1 + j].real   # unperturbed incident energy (index 0 is the bound state)
eta = 1e-6
lam = E + 1j * eta
 
psi_in = np.zeros(dim, dtype=complex)
psi_in[1 + j] = 1.0   # basis index (1+j) in the 0-photon block = continuum state E
 
rhs = (lam * np.eye(dim) - H0_full) @ psi_in   # = i*eta * psi_in, since psi_in is an H0 eigenstate
psi_lambda_exact = np.linalg.solve(lam * np.eye(dim) - H_full, rhs)
phi0_lambda_exact = psi_lambda_exact[0:M]      # the 0-photon component, i.e. |phi_0,lambda>
 
# ---------------------------------------------------------------------
# 4. Closed-form solution following the paper's eqs. (2)-(7) exactly
# ---------------------------------------------------------------------
IM = np.eye(M, dtype=complex)
 
 
def Gp(alpha):
    return np.linalg.inv(alpha * IM - Hp)
 
 
def Gp_tilde(alpha):
    # subtract the bound-state pole term: Gp(alpha) = |b><b|/(alpha+eps_b) + Gp_tilde(alpha)
    return Gp(alpha) - np.outer(phi_b, phi_b.conj()) / (alpha + eps_b)
 
 
# V^pol_lambda = sum_k f_k^dagger Gp_tilde(lambda - Omega_k) f_k   [corrected form, eq. in fixed .tex]
Vpol = np.zeros((M, M), dtype=complex)
for k in range(N):
    Vpol += np.abs(zeta[k])**2 * f @ Gp_tilde(lam - Omega[k]) @ f
 
Heff = Hp + Vpol
Geff = np.linalg.inv(lam * IM - Heff)
 
# direct-diffusion state: (lambda - Heff)|phi0^d> = (lambda - Hp)|psi_in>
rhs_M = np.zeros(M, dtype=complex)
rhs_M[1 + j] = 1j * eta   # (lambda - E)*e_{1+j} = i*eta*e_{1+j}, matches psi_in restricted to M-dim
phi0_d = np.linalg.solve(lam * IM - Heff, rhs_M)
 
# I(lambda,N) = sum_k |zeta_k|^2 / (lambda - Omega_k + eps_b)
I_lam = np.sum(np.abs(zeta)**2 / (lam - Omega + eps_b))
 
# X = <phi_b| f Geff f^dagger |phi_b>  =  Delta_eps(lambda) - i*gamma(lambda)/2
fdag_phib = f.conj().T @ phi_b   # = f @ phi_b since f Hermitian
X = phi_b.conj() @ f @ Geff @ fdag_phib
 
phib_f_phi0d = phi_b.conj() @ f @ phi0_d
 
phi0_lambda_closed = phi0_d + (Geff @ fdag_phib) * (phib_f_phi0d * I_lam) / (1 - I_lam * X)
 
# ---------------------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------------------
diff = np.linalg.norm(phi0_lambda_closed - phi0_lambda_exact)
ref = np.linalg.norm(phi0_lambda_exact)
rel_err = diff / ref
 
print("=" * 70)
print("Test: closed-form eq.(7) vs brute-force numerical solve of eqs.(1)")
print("=" * 70)
print(f"M (particle levels) = {M}, N (photon modes) = {N}")
print(f"incident channel energy E = {E:.6f}, eta = {eta:.1e}")
print(f"||phi_0,lambda||          (exact)  = {ref:.6e}")
print(f"||closed - exact||                 = {diff:.6e}")
print(f"relative error                     = {rel_err:.3e}")
print()
if rel_err < 1e-8:
    print("PASS: the closed-form resolvent solution (eqs. 2-7) reproduces the")
    print("      brute-force numerical solution to ~machine precision.")
    print("      => the algebra in Sec. 2.1 (bound-state pole extraction,")
    print("         V^pol, H^eff, Lippmann-Schwinger resummation) is correct.")
else:
    print("FAIL: closed-form and brute-force solutions disagree beyond")
    print("      numerical tolerance -- there is an algebra error in Sec. 2.1.")
 
# Also test at several other random incident channels / lambda points for robustness
print()
print("Robustness check over multiple random (channel, eta) draws:")
worst = 0.0
for trial in range(8):
    jj = rng.integers(0, M - 1)
    Ej = Hp[1 + jj, 1 + jj].real
    etaj = 10 ** rng.uniform(-8, -4)
    lamj = Ej + 1j * etaj
 
    def Gp_(alpha):
        return np.linalg.inv(alpha * IM - Hp)
 
    def Gpt_(alpha):
        return Gp_(alpha) - np.outer(phi_b, phi_b.conj()) / (alpha + eps_b)
 
    Vpol_j = np.zeros((M, M), dtype=complex)
    for k in range(N):
        Vpol_j += np.abs(zeta[k])**2 * f @ Gpt_(lamj - Omega[k]) @ f
    Heff_j = Hp + Vpol_j
    Geff_j = np.linalg.inv(lamj * IM - Heff_j)
    rhs_j = np.zeros(M, dtype=complex)
    rhs_j[1 + jj] = 1j * etaj
    phi0d_j = np.linalg.solve(lamj * IM - Heff_j, rhs_j)
    I_j = np.sum(np.abs(zeta)**2 / (lamj - Omega + eps_b))
    X_j = phi_b.conj() @ f @ Geff_j @ fdag_phib
    coup_j = phi_b.conj() @ f @ phi0d_j
    closed_j = phi0d_j + (Geff_j @ fdag_phib) * (coup_j * I_j) / (1 - I_j * X_j)
 
    psi_in_j = np.zeros(dim, dtype=complex)
    psi_in_j[1 + jj] = 1.0
    rhs_full_j = (lamj * np.eye(dim) - H0_full) @ psi_in_j
    exact_full_j = np.linalg.solve(lamj * np.eye(dim) - H_full, rhs_full_j)
    exact_j = exact_full_j[0:M]
 
    err = np.linalg.norm(closed_j - exact_j) / np.linalg.norm(exact_j)
    worst = max(worst, err)
    print(f"  channel {jj:2d}  E={Ej:6.3f}  eta={etaj:.1e}   rel.err = {err:.3e}")
 
print()
print(f"Worst-case relative error across trials: {worst:.3e}")
print("PASS" if worst < 1e-6 else "FAIL")

#!/usr/bin/env python3
"""
Colbert-Miller discrete-variable representation (DVR) for the radial
Schroedinger equation.
Reference: D. T. Colbert and W. H. Miller, J. Chem. Phys. 96, 1982 (1992),
grid on (-inf, inf) with uniform spacing dr (their eq. A6):
    T_ii = (pi^2 / 3)            / (2 mu dr^2)
    T_ij = 2 (-1)^(i-j) / (i-j)^2 / (2 mu dr^2),   i != j
Atomic units throughout (hbar = 1, energies in hartree, lengths in bohr).
The radial function is the reduced one, u(r) = r R(r); eigenvectors are
normalised so that sum_i |u_i|^2 dr = 1 (box normalisation).
Author:
Fausto Martelli
Institute of Complex Systems, CNR
Dept. of Physics
Sapienza University of Rome
"""
import numpy as np

# ------------------------------------------------------------------ constants (atomic units)
EH_EV = 27.211386245988          # hartree in eV
A0_A = 0.5291772109              # bohr in angstrom
CM1_EH = 219474.6313632          # cm^-1 per hartree
U_ME = 1822.888486209            # atomic mass unit in electron masses


def kinetic_dvr(n, dr, mu):
    """Colbert-Miller kinetic-energy matrix (n x n) on a uniform grid of spacing dr."""
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    with np.errstate(divide='ignore'):
        T = np.where(d == 0, np.pi ** 2 / 3.0,
                     2.0 * (-1.0) ** np.abs(d) / np.where(d == 0, 1, d) ** 2)
    return T / (2.0 * mu * dr ** 2)


def kinetic_fd3(n, dr, mu):
    """Three-point finite-difference kinetic-energy matrix (for comparison only)."""
    T = np.diag(np.full(n, 1.0 / (mu * dr ** 2)))
    off = np.full(n - 1, -1.0 / (2.0 * mu * dr ** 2))
    return T + np.diag(off, 1) + np.diag(off, -1)


def radial_eigenstates(V, r, mu, l=0, kinetic=kinetic_dvr):
    """
    Diagonalise H = T + V(r) + l(l+1)/(2 mu r^2) on the uniform grid r.

    V        : potential values on the grid (hartree)
    r        : uniform radial grid (bohr), r[0] > 0
    mu       : reduced mass (electron masses)
    l        : partial wave / rotational quantum number
    kinetic  : kinetic_dvr (default) or kinetic_fd3

    Returns (E, u): eigenvalues (hartree, ascending) and reduced radial
    eigenfunctions u[:, k] normalised to sum |u|^2 dr = 1.
    """
    dr = r[1] - r[0]
    Veff = V + l * (l + 1) / (2.0 * mu * r ** 2)
    H = kinetic(len(r), dr, mu) + np.diag(Veff)
    E, u = np.linalg.eigh(H)
    u /= np.sqrt(np.sum(u ** 2, axis=0) * dr)
    return E, u


# ------------------------------------------------------------------ CH(X2Pi) Morse test
def ch_morse():
    """Morse curve fixed by Re = 1.121 A, we = 2860.4118 cm^-1, De = 3.604 eV."""
    mu = 12.0000 * 1.007825 / (12.0000 + 1.007825) * U_ME
    Re, De = 1.121 / A0_A, 3.604 / EH_EV
    we = 2860.4118 / CM1_EH
    a = we * np.sqrt(mu / (2.0 * De))
    wexe = we ** 2 / (4.0 * De)

    def V(r):
        return De * (1.0 - np.exp(-a * (r - Re))) ** 2 - De

    def E_exact(v):
        return -De + we * (v + 0.5) - wexe * (v + 0.5) ** 2

    return mu, V, E_exact


if __name__ == '__main__':
    mu, V, E_exact = ch_morse()
    r = np.linspace(0.6, 60.0, 1415)            # dr = 0.0420 a0, as in the paper
    print(f"grid: {len(r)} points, dr = {r[1]-r[0]:.4f} a0, r = {r[0]}-{r[-1]} a0")
    v = np.arange(20)
    Eex = E_exact(v)
    Ed, _ = radial_eigenstates(V(r), r, mu, kinetic=kinetic_dvr)
    Ef, _ = radial_eigenstates(V(r), r, mu, kinetic=kinetic_fd3)
    print(f"bound levels found: DVR {np.sum(Ed < 0)}, FD3 {np.sum(Ef < 0)}")
    print("  v   E_exact (eV)   |FD3 - exact| (meV)   |DVR - exact| (meV)")
    for k in v:
        print(f" {k:2d}   {Eex[k]*EH_EV:11.6f}   {abs(Ef[k]-Eex[k])*EH_EV*1e3:18.4f}"
              f"   {abs(Ed[k]-Eex[k])*EH_EV*1e3:18.2e}")

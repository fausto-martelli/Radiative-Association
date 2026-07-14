# Radiative-Association

Numerical validation code for "Quantum-Regime Radiative Association: An Analytical Approach"

This is a set of four standalone Python scripts written to independently check the
analytical derivation in the paper (a resolvent/Fano-Feshbach treatment of radiative
association) and, as far as possible, its application to a real system, C + H → CH + ħω.
They correspond to the "Numerical tests" section of the manuscript.

None of the scripts import from one another — each is self-contained and can be run on
its own with python3 <script>.py.

Requirements


Python 3, numpy (all scripts)
sympy (only verify_hbar_pi_corrections.py)


No other third-party packages are needed. scipy is deliberately avoided (it wasn't
available in the sandbox this was developed in); verify_hbar_pi_corrections.py has a
small hand-rolled numerical integrator in place of scipy.integrate.quad.

Typical runtime: a few seconds each, except validate_CH_radiative_association.py and
reproduce_jones2022.py, which diagonalize a 3000×3000 grid Hamiltonian and take
5-10 seconds.

Recommended reading/run order


test_resolvent_algebra.py
verify_hbar_pi_corrections.py
validate_CH_radiative_association.py
reproduce_jones2022.py


Each stage builds on the physical picture of the one before it, but not on its code.


1. test_resolvent_algebra.py

What it checks: the algebra of section 2.1 of the paper — specifically, that the
closed-form resolvent solution (Eq. 7, obtained by extracting the bound-state pole of
the free Green's function and resumming a Lippmann-Schwinger-type equation) is an
exact solution of the original coupled linear system (Eq. 1), not just an
approximation.

Method: builds a toy model — one bound state plus M-1 discretized "continuum"
levels, coupled to N bosonic (photon) modes via a random Hermitian operator f — and
solves the same scattering problem two independent ways:


Brute force: assemble the full (1+N)M × (1+N)M Hamiltonian explicitly (all
0-photon and 1-photon sectors) and solve (λ - H)ψ = (λ - H⁰)ψ_in by direct matrix
inversion.
Closed form: evaluate Eq. 7 symbol-by-symbol (bound-state pole extraction,
polarization potential, effective Hamiltonian, resonance denominator) using the same
inputs.


It then compares the two |φ_0,λ⟩ solutions and repeats this for 8 random
incident channels / regularization parameters η.

Result: relative error ~1e-14 to 1e-20 (machine precision) in every trial. This
is the main evidence that the pole-extraction/resummation derivation in the paper is
algebraically correct, independent of any physical interpretation.


2. verify_hbar_pi_corrections.py

What it checks: two specific numerical-prefactor corrections made to the paper
during review (an ℏ power in the Sokhotski–Plemelj residue used in Sec. 2.2, and a
π-power/coefficient in the cross-section substitution, Eq. 10 of Sec. 2.4).

Method, Part 1 (ℏ check): numerically integrates the Lorentzian-regularized
integral defining Im[I(E)] for decreasing regularization width η, with ℏ set to a
non-trivial value (0.7, not 1) specifically so that a missing/extra power of ℏ would be
visible as a numerical mismatch rather than hiding inside ℏ=1 units. Compares the
η→0 limit against two candidate closed-form residues (the original vs. the corrected
version).

Method, Part 2 (π check): uses sympy to symbolically substitute
k̄ = 2πν̄/c and p² = 2mE into the cross-section expression and checks which of two
candidate final forms (32π⁵ vs. 128π⁶) the substitution actually produces.

Result: the corrected forms are confirmed in both cases — the ℏ→0 numerical limit
matches the corrected residue (not the original, which is off by exactly one factor of
ℏ), and the symbolic substitution reproduces 128π⁶/(3c³mE) exactly.


3. validate_CH_radiative_association.py

What it checks: whether the paper's formalism holds up (Part A/B) and gives
physically sensible numbers (Part C) when applied to a real molecule rather than a toy
model.

Part A — real potential. Builds a Morse potential for CH(X²Π) from literature
spectroscopic constants (Rₑ, ωₑ, ωₑχₑ, Dₑ; see the paper's Sec. 3.2 for sources),
diagonalizes it on a 3000-point radial grid, and compares the numerical v=0–3
vibrational levels against the analytic Morse term-value formula as a sanity check on
the discretization (agreement: 0.3 meV at v=0, growing to ~15 meV by v=3, as expected
for a finite-difference grid).

Part B — algebra test on the real system. Repeats the test from
test_resolvent_algebra.py, but with H_p now built from the real CH potential (Part
A) and a dipole-like coupling operator computed from the corresponding grid
wavefunctions, instead of random matrices. Relative error: ~7e-12.

Part C — cross section and rate coefficient. Converts box-normalized
bound/continuum wavefunctions to energy normalization via the local density of states,
evaluates the paper's cross-section formula (Eq. 10), and thermally averages it
(Eq. k(T)) to estimate k(20 K) and k(100 K) for the direct (non-resonant),
single-surface X²Π→X²Π channel, using an illustrative (not ab initio) dipole moment
function.

Result: k(20 K) ≈ 4×10⁻²² cm³/s, ~5 orders of magnitude below the real literature
value for the full C+H→CH reaction (~8×10⁻¹⁷ cm³/s, Jones et al. 2022) — expected,
since this calculation deliberately omits the resonance-capture mechanism that
literature studies show dominates at these temperatures (see script 4).


4. reproduce_jones2022.py

What it checks: how much of the gap identified in script 3 can be closed by using
real literature inputs (potential curves, dipole-moment decay constants) rather than
a single generic model potential/dipole, and how much is irreducibly missing without
external resonance data.

Method: rebuilds the calculation of script 3's Part C for two of the three physical
formation channels identified in Jones, Gustafsson & Nyman (2022, MNRAS 517, 4892):


X²Π → X²Π (direct), using the real well depth (3.604 eV) and the real X²Π
permanent-dipole decay constant, with the amplitude anchored to the commonly-quoted
equilibrium value (~1.46 D — confirmed independently via Baluja & Msezane 2001,
J. Phys. B 34, 3157).
B²Σ⁻ → X²Π (direct + shape resonances), using a potential built to match the real
well depth (0.372 eV) and barrier height (0.128 eV) from Billoux, Cressault & Gleizes
(2014, JQSRT 133, 434), with a transition-dipole decay constant from the literature
but an illustrative (unsourced) amplitude — flagged explicitly in the script's output.


The third, dominant channel (resonance capture via inverse predissociation through the
A²Δ state) is not computed — it requires radiative and predissociation linewidths
from an external program/line list not available in this environment — and is instead
represented by the range reported by earlier studies that considered that channel in
isolation (Julienne & Krauss 1973; Brzozowski et al. 1976, as quoted in Jones et al.
2022).

Result: combining the two computed channels with the cited range for the missing
one gives k(100 K) ≈ 5×10⁻²⁰ to 1×10⁻¹⁷ cm³/s, versus the real total of 3.5×10⁻¹⁷
cm³/s — within a factor of a few to a few hundred, with essentially all the remaining
gap attributable specifically to the uncomputed resonance channel rather than a general
failure of the direct-channel formalism.


Caveats (read before citing these numbers anywhere)


Dipole moments are not, in general, ab initio. Where a real amplitude was
available and confirmed (the CH X²Π permanent dipole, ~1.46 D) it was used; where it
wasn't (the B²Σ⁻–X²Π transition dipole amplitude), an order-of-magnitude estimate was
used instead and is explicitly labeled as such in both the script output and the
paper.
The B²Σ⁻ barrier potential is a hand-built approximation (an attractive Morse well
plus an analytic bump function tuned to hit the literature well depth and barrier
height), not the actual RKR potential from Billoux et al. (2014).
The dominant CH-formation channel (A²Δ inverse predissociation) is not computed
anywhere in this code. It requires non-adiabatic/predissociation coupling data this
environment has no access to. Script 4's "combined" estimate substitutes a literature
range for that piece rather than deriving it.
These scripts test the algebra of the paper rigorously (scripts 1-3B are exact,
reproducible, and require no external data). They test the physical plausibility of
the resulting formulas illustratively (scripts 3C-4), not quantitatively.

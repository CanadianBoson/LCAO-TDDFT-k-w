# LCAO-TDDFT-k-ω
LCAO TDDFT for GPAW in frequency space

This module defines an LCAOTDDFTq0 class
which implements the LCAO mode TDDFT-ω
implementation in the optical limit defined in:

[Glanzmann, L. N.; Mowbray, D. J.; del Valle, D. G. F.; Scotognella, F.; Lanzani, G.; Rubio, A. *J. Phys. Chem. C* 2015, **120**, 1926--1935.](http://dx.doi.org/10.1021/acs.jpcc.5b10025 "doi:10.1021/acs.jpcc.5b10025")

Linear Combination of Atomic Orbitals (LCAO) mode
Time Dependent Density Functional Theory (TDDFT)
in the frequency domain and  the optical limit q → 0⁺
neglecting local crystal field effects.

Parallelization over k-points, spin, and domain are implemented.
Singlet calculations require parallelization to not be over spin.
Supports use of ScaLAPACK which requires initialization
of the lower triangle of the gradient of phi matix.

The lcao_tddft_q0.py script may be either executed directly
from the command line or loaded as a Python module.
For help on the command line interface try:

$ lcao-tddft-k-omega --help

Exciton Density Calculation

This module defines an ExcitonDensity class
which calculates the spatially and energetically
resolved electron and hole densities
ρₑ and ρₕ for a given energy ω based on the 
transition n  → m intensities,  i.e., 
the square magnitude of the oscillator strengths 
|fₙₙₖ|², obtained from the LCAOTDDFTq0 class, 
and the square of the Kohn-Sham wave functions 
ψₙₖ(rₕ) and ψₘₖ(rₑ) at a given k-point,
based on the two-point excitonic density

ρₑₓ(rₑ,rₕ,ω) = ΣₙₘΣₖwₖ|fₙₘₖ|²|ψₙₖ(rₕ)|²|ψₘₖ(rₑ)|² η²/((ω-εₘₖ+εₙₖ)²+η²)

with the electron and hole densities obtained from ρₑₓ(rₑ,rₕ,ω)
by integrating w.r.t. rₕ and rₑ respectively, i.e.,

ρₑ(rₑ,ω) = ∫ρₑₓ(rₑ,rₕ,ω)drₕ
         = ΣₙₘΣₖwₖ|fₙₘₖ|²|ψₘₖ(rₑ)|² η²/((ω-εₘₖ+εₙₖ)²+η²)

ρₕ(rₕ,ω) = ∫ρₑₓ(rₑ,rₕ,ω)drₑ
         = ΣₙₘΣₖwₖ|fₙₘₖ|²|ψₙₖ(rₕ)|² η²/((ω-εₘₖ+εₙₖ)²+η²)
so that

Im[ε(ω)] = ∬ρₑₓ(rₑ,rₕ,ω)drₕdrₑ = ∫ρₑ(rₑ,ω)drₑ = ∫ρₕ(rₕ,ω)drₕ

The exciton_density.py script may be either executed directly
from the command line or loaded as a Python module.
For help on the command line interface try:

$ exciton-density --help

Installation:

$ gpaw-python setup.py install

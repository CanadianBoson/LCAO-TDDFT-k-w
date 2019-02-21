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

This module defines a ExcitonDensity class
which calculates the electron and hole densities
at a given energy omega based on the transition
intensities obtained from the LCAOTDDFTq0 class

ρₑ(ω) = Σₙₙ' fₙₙ' |ψₙ'|² exp(-(ω-(εₙ-εₙ'))²/2σ²)/σ√2π)
ρₕ(ω) = Σₙₙ' fₙₙ' |ψₙ |² exp(-(ω-(εₙ-εₙ'))²/2σ²)/σ√2π)
where fₙₙ' is the intensity of the n → n' transition


The exciton_density.py script may be either executed directly
from the command line or loaded as a Python module.
For help on the command line interface try:

$ exciton-density --help

Installation:

$ gpaw-python setup.py install

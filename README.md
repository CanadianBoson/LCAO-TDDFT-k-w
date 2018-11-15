# LCAO-TDDFT-k-ω
LCAO TDDFT for GPAW in frequency space

This module defines an LCAOTDDFTq0 class
which implements the LCAO mode TDDFT-ω
implementation in the optical limit defined in:

[Glanzmann, L. N.; Mowbray, D. J.; del Valle, D. G. F.; Scotognella, F.; Lanzani, G.; Rubio, A. *J. Phys. Chem. C* 2015, **120**, 1926--1935.](http://dx.doi.org/10.1021/acs.jpcc.5b10025 "doi:10.1021/acs.jpcc.5b10025")

Linear Combination of Atomic Orbitals (LCAO) mode
Time Dependent Density Functional Theory (TDDFT)
in the frequency domain and  the optical limit q→0⁺
neglecting local crystal field effects.

Parellization over k-points, spin, and domain are implemented.
Singlet calculations require parallelization to not be over spin.

Installation:

gpaw-python setup.py install

The lcao_tddft_q0.py script may be either executed directly
from the command line or loaded as a Python module.
For help on the command line interface try:

$ lcao-tddft-k-omega --help


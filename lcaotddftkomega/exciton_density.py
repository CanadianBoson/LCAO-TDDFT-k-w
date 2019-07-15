#!/usr/bin/env gpaw-python
# -*- coding: utf-8 - *-

"""This module defines an ExcitonDensity class
which calculates the spatially and energetically
resolved electron and hole charge densities
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
ρₑ(rₑ,ω) = -∫ρₑₓ(rₑ,rₕ,ω)drₕ
         = -ΣₙₘΣₖwₖ|fₙₘₖ|²|ψₘₖ(rₑ)|² η²/((ω-εₘₖ+εₙₖ)²+η²)
ρₕ(rₕ,ω) = ∫ρₑₓ(rₑ,rₕ,ω)drₑ
         = ΣₙₘΣₖwₖ|fₙₘₖ|²|ψₙₖ(rₕ)|² η²/((ω-εₘₖ+εₙₖ)²+η²)
so that
Im[ε(ω)] = ∬ρₑₓ(rₑ,rₕ,ω)drₕdrₑ = -∫ρₑ(rₑ,ω)drₑ = ∫ρₕ(rₕ,ω)drₕ
"""
from numpy import array, zeros
from gpaw import GPAW
from ase.io.cube import write_cube

class ExcitonDensity(object):
    """Class for calculating the spatially and energetically
    resolved electron and hole densities, ρₑ(rₑ,ω) and ρₕ(rₕ,ω)
    based on the transition intensitiess obtained from the
    LCAOTDDFTq0 class, where Im[ε(ω)] = ∫ρₑ(rₑ,ω)drₑ = ∫ρₕ(rₕ,ω)drₕ"""

    def __init__(self,
                 calc,
                 omega,
                 transitions,
                 eta=0.1,
                 cutoff=1e-6,
                 axesdir=None,
                 eels_prefactor=1.0):
        """Creates an ExcitonDensity objtect.

        calc		GPAW LCAO calculator or gpw filename
        omega     	Energy to calculate exciton density in eV
        transitions	Transition intensities from LCAOTDDFTq0
        eta     	Lorentzian broadening (0.1 eV)
        cutoff		Cutoff for including transitions (1e-6)
        axesdir		Direction of excitations to include
        eels_prefactor  Prefactor for EELS
        """

        if not isinstance(calc, GPAW):
            calc = GPAW(calc, txt=None)
        if calc.wfs.mode != 'lcao':
            raise TypeError('Calculator is not for an LCAO mode calculation!')
        self.calc = calc
        self.omega = omega
        self.calculated = False
        self.set_energy(omega)
        self.transitions = transitions
        self.eta = eta
        self.axesdir = axesdir
        self.set_excitation_direction(axesdir)
        if isinstance(transitions, str):
            self.read_transitions(transitionsfile=transitions)
        # Initialize electron and hole densities
        n_c = self.calc.get_number_of_grid_points()
        self.rho_e = zeros(n_c, dtype=float)
        self.rho_h = zeros(n_c, dtype=float)
        self.cutoff = cutoff
        self.eels_prefactor = eels_prefactor

    def read_transitions(self, transitionsfile):
        """Read transitions from a LCAOTDDFTq0 file"""

        transitionsdata = open(transitionsfile, 'r').readlines()
        axesdirs = {'x': 0, 'y': 1, 'z': 2}
        transitionslist = []
        for i in range(1, len(transitionsdata)):
            tdata = array(transitionsdata[i].strip('[').split(']')[0].split(','),
                          dtype=float)
            e_n, f_nn = tdata[:2]
            i_n, j_n, s_n, kpt_n = array(tdata[2:], dtype=int)
            axes_n = axesdirs[transitionsdata[i].split()[-1]]
            transitionslist.append([e_n,
                                    f_nn,
                                    i_n,
                                    j_n,
                                    s_n,
                                    kpt_n,
                                    axes_n])
        self.transitions = transitionslist

    def set_energy(self, energy):
        """Specify energy for calculating exciton density"""
        self.omega = energy
        self.calculated = False

    def set_excitation_direction(self, axesdir):
        """Specify energy for calculating exciton density"""
        if axesdir != self.axesdir:
            self.axesdir = axesdir
            self.calculated = False

    def get_prefactor(self, energy, intensity):
        """Calculate prefactor for a transition
        n  → m at a given k-point and energy ω
        P = wₖ|fₙₙₖ|² η² / ((ω-εₘₖ+εₙₖ)²+η²)"""
        prefactor = intensity * self.eta**2
        prefactor /= (self.omega - energy)**2 + self.eta**2
        prefactor *= self.eels_prefactor
        return prefactor

    def calculate(self, recalculate=False):
        """Calculate the exciton density"""
        if not self.calculated or recalculate:
            self.rho_e *= 0
            self.rho_h *= 0
            for transition in self.transitions:
                e_n, f_nn, i_n, j_n, s_n, kpt_n, axes_n = transition
                if self.axesdir is None or self.axesdir == axes_n:
                    prefactor = self.get_prefactor(e_n, f_nn)
                    if prefactor > self.cutoff:
                        self.add_densities(prefactor,
                                           i_n,
                                           j_n,
                                           s_n,
                                           kpt_n)
            self.calculated = True
        return

    def add_densities(self, prefactor, i_n, j_n, s_n, kpt_n):
        """Add prefactor times wave function densities
        for the given spin and k-point to the electron and hole
        densities

        prefactor	Prefactor
        i_n		Index of hole wave function
        j_n		Index of electron wave function
        s_n		Spin channel
	kpt_n		k-point of transition"""
        # Update hole density
        psi = self.calc.get_pseudo_wave_function(i_n, kpt_n, s_n)
        rho = (psi * psi.conj()).real
        self.rho_h += prefactor * rho / rho.sum()
        # Update electron density
        psi = self.calc.get_pseudo_wave_function(j_n, kpt_n, s_n)
        rho = (psi * psi.conj()).real
        self.rho_e -= prefactor * rho / rho.sum()

    def write_densities(self, outfilename):
        """Write electron and hole densities to cube files

        outfilename	Output file name"""

        self.calculate()
        name = outfilename+'_rho_e.cube'
        write_cube(open(name, 'w'),
                   self.calc.get_atoms(),
                   data=self.rho_e)
        name = outfilename+'_rho_h.cube'
        write_cube(open(name, 'w'),
                   self.calc.get_atoms(),
                   data=self.rho_h)


def read_arguments():
    """Input Argument Parsing"""

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='name of input GPAW file')
    parser.add_argument('transitionfilename', type=str,
                        help='name of transition file')
    parser.add_argument('-w', '--omega',
                        type=float,
                        help='energy of transitions')
    parser.add_argument('-kBT', '--eta',
                        help='electronic temperature (%(default)s eV)',
                        default=0.1, type=float)
    parser.add_argument('-eels', '--eels_prefactor',
                        help='eels prefactor', default=1.0, type=float)
    return parser.parse_args()

def main():
    """Command Line Executable"""
    # Read Arguments
    args = read_arguments()
    exciton = ExcitonDensity(args.filename,
                             args.omega,
                             args.transitionfilename,
                             args.eta,
                             args.eels_prefactor)
    axes = {0: 'x', 1: 'y', 2: 'z'}
    for direction in range(3):
        exciton.set_excitation_direction(direction)
        outfilename = args.filename.split('.gpw')[0]+'_'+axes[direction]+'_'+str(args.omega)+'eV_EELS_'+str(args.eels_prefactor)
        exciton.write_densities(outfilename)

if __name__ == '__main__':
    main()

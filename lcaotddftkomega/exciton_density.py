#!/usr/bin/env gpaw-python
# -*- coding: utf-8 - *-

"""This module defines a ExcitonDensity class
which calculates the electron and hole densities
at a given energy omega based on the transition
intensities obtained from the LCAOTDDFTq0 class

ρₑ(ω) = Σₙₙ' fₙₙ' |ψₙ'|² exp(-(ω-(εₙ-εₙ'))²/2σ²)/σ√2π)
ρₕ(ω) = Σₙₙ' fₙₙ' |ψₙ |² exp(-(ω-(εₙ-εₙ'))²/2σ²)/σ√2π)

where fₙₙ' is the intensity of the n→n' transition
"""
from numpy import array
from gpaw import GPAW

class ExcitonDensity(object):
    """Class for calculating electron and hole densities
    based on the transition intensitiess obtained from the
    LCAOTDDFTq0 class"""

    def __init__(self,
                 calc,
                 omega,
                 transitions):
        """Creates an ExcitonDensity objtect.

        calc		GPAW LCAO calculator or gpw filename
        omega     	Energy to calculate exciton density in eV
        transitions	Transition intensities from LCAOTDDFTq0
        """

        if not isinstance(calc, GPAW):
            calc = GPAW(calc, txt=None)
        if calc.wfs.mode != 'lcao':
            raise TypeError('Calculator is not for an LCAO mode calculation!')
        self.calc = calc
        self.omega = omega
        self.transitions = transitions
        self.calculated = False
        if isinstance(transitions, str):
            self.read_transitions(transitionsfile=transitions)

    def read_transitions(self, transitionsfile):
        """Read transitions from a LCAOTDDFTq0 file"""

        transitionsdata = open(transitionsfile, 'r').readlines()
        transitionslist = []
        for i in range(1, len(transitionsdata)):
            e_n, f_nn = array(transitionsdata[i].split()[:2], dtype=float)
            i_n = int(transitionsdata[i].split()[2])
            j_n, kpt = array(transitionsdata[i].split()[4:], dtype=int)
            transitionslist.append([e_n, f_nn, i, j, kpt])
        self.transitions = transitionslist

    def set_energy(self, energy):
        """Specify energy for calculating exciton density"""
        self.omega = energy
        self.calculated = False

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
    return parser.parse_args()

def main():
    """Command Line Executable"""
    # Read Arguments
    args = read_arguments()
    exciton = ExcitonDensity(args.filename, args.omega, args.transitionfilename)


if __name__ == '__main__':
    main()

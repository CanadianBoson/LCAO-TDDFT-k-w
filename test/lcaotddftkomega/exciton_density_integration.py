#!/usr/bin/env gpaw-python
# -*- coding: utf-8 - *-

"""Test for the exciton_density module"""
import sys
from numpy import array, zeros, arange
from gpaw import GPAW
from lcaotddftkomega.exciton_density import ExcitonDensity
from ase.io.cube import write_cube

class IntegrateExcitonDensity(ExcitonDensity):
    """Class for integrating the exciton density to obtain the spectra"""

    def add_densities(self, prefactor, i_n, j_n, s_n, kpt_n):
        """Add prefactor times wave function densities
        for the given spin and k-point to the electron and hole
        densities

        prefactor	Prefactor
        i_n		Index of hole wave function
        j_n		Index of electron wave function
        s_n		Spin channel
	kpt_n		k-point of transition"""
        self.rho_h += prefactor
        self.rho_e -= prefactor

    def write_densities(self, outfilename):
        """Write electron and hole densities to cube files

        outfilename	Output file name"""

        self.calculate()
        f = open(outfilename+'.dat', 'a')
        print >> f, self.omega, self.rho_h[0][0][0]
        f.close()

def read_arguments():
    """Input Argument Parsing"""

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='name of input GPAW file')
    parser.add_argument('transitionfilename', type=str,
                        help='name of transition file')
    parser.add_argument('-wmin', '--omegamin',
                        help='energy range minimum (%(default)s eV)',
                        default=0., type=float)
    parser.add_argument('-wmax', '--omegamax',
                        help='energy range maximum (%(default)s eV)',
                        default=10., type=float)
    parser.add_argument('-dw', '--domega',
                        help='energy increment (%(default)s eV)',
                        default=0.025, type=float)
    parser.add_argument('-kBT', '--eta',
                        help='electronic temperature (%(default)s eV)',
                        default=0.1, type=float)
    parser.add_argument('-ct', '--cutoff',
                        help='cutoff for including transitions (%(default)s)',
                        default=0., type=float)
    return parser.parse_args()

def main():
    """Command Line Executable"""
    # Read Arguments
    args = read_arguments()
    exciton = IntegrateExcitonDensity(args.filename,
                                      0.0,
                                      args.transitionfilename,
                                      args.eta,
                                      args.cutoff)
    axes = {0: 'x', 1: 'y', 2: 'z'}
    for omega in arange(args.omegamin, args.omegamax, args.domega):
        exciton.set_energy(omega)
        for direction in range(3):
            exciton.set_excitation_direction(direction)
            outfilename = args.transitionfilename.split('.dat')[0]+'_'+axes[direction]
            exciton.write_densities(outfilename)

if __name__ == '__main__':
    main()

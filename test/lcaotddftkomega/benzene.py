#!/usr/bin/env gpaw-python

"""Test for domain parallelization within
LCAOTDDFTq0 class using the optical absorption
spectra of benzene

Peak at ~ 5.2 eV of ~ 3 e^2/hbar
Peak at ~ 9 eV of ~ 0.24 e^2/hbar"""
from __future__ import print_function
from os.path import exists
from ase.build import molecule
from gpaw.test import equal
from gpaw import GPAW, FermiDirac
from gpaw import __version__ as gpaw_version
from gpaw.mpi import world
from lcaotddftkomega.lcao_tddft_q0 import LCAOTDDFTq0

def main():
    """Benzene LCAO-TDDFT-k-omega test
    LDA, dzp, 1x1x1 k-point sampling"""

    xcf = 'LDA'
    basis = 'dzp'
    kpt = 1
    name = 'benzene_'+xcf+'_'+basis+'_'+str(kpt)+'x'+str(kpt)+'.gpw'
    if not exists(name):
        atoms = molecule('C6H6')
        atoms.center(4)
        atoms.set_pbc(True)
        calc = GPAW(mode='lcao',
                    xc=xcf,
                    h=0.26,
                    basis=basis,
                    kpts=[kpt, kpt, kpt],
                    occupations=FermiDirac(width=0.001))
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write(name, mode='all')
        del calc
    lcao_tddft = LCAOTDDFTq0(name, eta=0.2, verbose=True)
    epsilon = lcao_tddft.get_epsilon()
    lcao_tddft.write_dielectric_function(name.split('.gpw')[0])
    if world.rank == 0:
        energies = epsilon[0]
        im_epsilon = epsilon[2].sum(axis=0)
        print("emax =", energies[im_epsilon.argmax()])
        print("im_epsilonmax =", im_epsilon.max())
        print("im_epsilon_9 =", im_epsilon[360])
        emax = 5.2
        if gpaw_version < '1.5.0':
            im_epsilonmax = 3.00301106687
            im_epsilon_9 = 0.236562254483
        else:
            im_epsilonmax = 3.0011048853480093
            im_epsilon_9 = 0.22906754250253078
        equal(emax, energies[im_epsilon.argmax()], 1e-8)
        equal(im_epsilonmax, im_epsilon.max(), 1e-8)
        equal(im_epsilon_9, im_epsilon[360], 1e-8)

main()

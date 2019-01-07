#!/usr/bin/env gpaw-python

"""Test for k-point parallelization within
LCAOTDDFTq0 class using the optical absorption
spectra of graphene

Peak at ~ 4 eV of ~ 3.4 e^2/hbar
Flat region at ~ 1 eV of ~ 1 e^2/hbar"""
from __future__ import print_function
from os.path import exists
from ase import Atom, Atoms
from ase.units import Bohr
from gpaw.test import equal
from gpaw import GPAW, FermiDirac
from gpaw.mpi import world
from lcaotddftkomega.lcao_tddft_q0 import LCAOTDDFTq0

def main():
    """Graphene LCAO-TDDFT-k-omega test
    LDA, szp(dzp), 71x71x1 k-point sampling"""

    xcf = 'LDA'
    basis = 'szp(dzp)'
    kpt = 71
    name = 'graphene_'+xcf+'_'+basis+'_'+str(kpt)+'x'+str(kpt)+'.gpw'
    if not exists(name):
        # Graphene Cell Parameter in Bohr
        acell = 4.651 * Bohr
        cell = [[acell, 0, 0],
                [-0.5*acell, 3**0.5/2.*acell, 0],
                [0, 0, 8.32]]

        atoms = Atoms()
        atoms.append(Atom('C', [0.5*acell, -acell/(2*3**0.5), 0]))
        atoms.append(Atom('C', [0.5*acell, acell/(2*3**0.5), 0]))
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        calc = GPAW(mode='lcao',
                    xc=xcf,
                    h=0.26,
                    basis=basis,
                    kpts=[kpt, kpt, 1],
                    occupations=FermiDirac(width=0.001),
                    parallel={'sl_auto': True},)
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write(name, mode='all')
        del calc
    lcao_tddft = LCAOTDDFTq0(name, eta=0.2, verbose=True)
    sigma = lcao_tddft.get_sigma()
    lcao_tddft.write_optical_conductivity(name.split('.gpw')[0])
    lcao_tddft.write_polarization(name.split('.gpw')[0], '2D')
    if world.rank == 0:
        re_sigma2d = (sigma[1][0] + sigma[1][1])*2
        energies = sigma[0]
        print("emax =", energies[re_sigma2d.argmax()])
        print("re_sigmamax =", re_sigma2d.max())
        print("re_sigma_1 =", re_sigma2d[40])
        emax = 4.0
        re_sigmamax = 3.44
        re_sigma_1 = 1.06
        equal(emax, energies[re_sigma2d.argmax()], 1e-8)
        equal(re_sigmamax, re_sigma2d.max(), 0.05)
        equal(re_sigma_1, re_sigma2d[40], 0.05)

main()

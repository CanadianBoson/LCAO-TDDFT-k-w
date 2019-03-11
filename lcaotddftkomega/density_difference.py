#!/usr/bin/env gpaw-python
# -*- coding: utf-8 - *-

"""Calculate electron-hole charge density
Δρ(r,ω) = |ρₕ(r,ω)| - |ρₑ(r,ω)| = ρₕ(r,ω) + ρₑ(r,ω)
where ∫Δρ(r,ω)dr = 0"""

from sys import argv
from ase.io.cube import read_cube_data, write_cube

def main():
    """Command line executable"""
    if not argv[1].endswith('.cube'):
        raise ValueError(argv[1])

    rho_e, atoms = read_cube_data(open(argv[1], 'r'))

    if not argv[2].endswith('.cube'):
        raise ValueError(argv[2])

    rho_h, atoms = read_cube_data(open(argv[2], 'r'))
    drho = rho_h + rho_e
    name = argv[1].split('.cube')[0]+'drho.cube'
    write_cube(open(name, 'w'), atoms, data=drho)

if __name__ == '__main__':
    main()

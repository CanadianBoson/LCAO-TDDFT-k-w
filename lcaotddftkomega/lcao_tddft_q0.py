#!/usr/bin/env gpaw-python

"""This module defines an LCAOTDDFTq0 class
which implements the LCAO mode TDDFT-omega
implementation in the optical limit defined in:

[1] Glanzmann, L. N.; Mowbray, D. J.; del Valle, D. G. F.; Scotognella, F.;
Lanzani, G.; Rubio, A. J. Phys. Chem. C 2015, 120, 1926--1935.

Linear Combination of Atomic Orbitals (LCAO) mode
Time Dependent Density Functional Theory (TDDFT)
in the frequency domain and  the optical limit q -> 0+
neglecting local crystal field effects.

Parellization over k-points, spin, and domain are implemented.
Singlet calculations require parallelization to not be over spin.
Supports use of ScaLAPACK which requires initialization
of the lower triangle of the gradient of phi matix.

The lcao_tddft_q0.py script may be either executed directly
from the command line or loaded as a Python module.
For help on the command line interface try:

$ lcao_tddft_q0.py --help"""


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from math import pi
from numpy import empty, zeros, ones, identity
from numpy import dot, cross, outer, arange, array, mod
from numpy.lib.twodim_base import triu_indices
from gpaw import GPAW
from gpaw import __version__ as gpaw_version
from gpaw.mpi import world
from gpaw.utilities.blas import gemm
from ase.units import create_units, __codata_version__
from ase.parallel import parprint
HA = create_units(__codata_version__)['Hartree']


class LCAOTDDFTq0(object):
    """Class for performing LCAO TDDFT in the optical limit q -> 0+"""

    def __init__(self,
                 calc,
                 eta=0.1,
                 scissor=0.0,
                 cutocc=1e-5,
                 verbose=False,
                 scalapack=False):
        """Creates a LCAOTDDFTq0 object.

        calc		GPAW LCAO calculator or gpw filename
        eta     	Lorentzian broadening (0.1 eV)
        cutocc		cutoff for occupance in [0,0.5)
        verbose		Print verbose output (False)
        scalapack	Use ScaLAPACK (False)
        """

        if not isinstance(calc, GPAW):
            if verbose:
                txt = '-'
            else:
                txt = None
            calc = GPAW(calc,
                        txt=txt,
                        parallel={
                            'sl_auto': scalapack,
                            'augment_grids': True,
                            'kpt': None, 'domain': None,
                            'band': 1})
        if calc.wfs.mode != 'lcao':
            raise TypeError('Calculator is not for an LCAO mode calculation!')
        self.calc = calc
        self.nocc = int(calc.get_number_of_electrons() +
                        calc.parameters['charge']) // 2
        # unit cell in Bohr^3
        cell = self.calc.wfs.gd.cell_cv
        # unit cell volume in Bohr^3
        volume = abs(dot(cell[0], cross(cell[1], cell[2])))
        # prefactor for overlap_nm in chi0
        self.prefactor = - 4 * pi / volume
        self.be_verbose(verbose)
        self.use_singlet(False)
        self.use_hilbert_transform(False)
        self.calculated = False
        self.set_energy_range()
        self.calculate_transitions(False)
        self.verboseprint('Electronic Temperature', eta, 'eV')
        self.verboseprint('|f_n - f_m| >', cutocc)
        self.verboseprint('Initializing Positions')
        calc.initialize_positions(calc.get_atoms())
        self.cutocc = cutocc
        self.eta = eta / HA
        self.scissor = scissor / HA
        self.verboseprint('Calculating Basis Function Gradients')
        self.grad_phi_kqvnumu = self.get_grad_phi()

    def set_energy_range(self, omegamin=0.0, omegamax=10.0, domega=0.025):
        """Set the energy range in eV
        omegamin	minimum energy
        omegamax	maximum energy
        domega		energy spacing"""
        self.verboseprint('Setting energy range from',
                          omegamin, 'to', omegamax,
                          'eV in increments of ', domega, 'eV')
        omega_w = arange(omegamin, omegamax, domega)
        self.omega_w = omega_w / HA
        if world.rank is 0:
            self.re_epsilon_qvw = ones([3, len(self.omega_w)])
        else:
            self.re_epsilon_qvw = zeros([3, len(self.omega_w)])
        self.im_epsilon_qvw = zeros([3, len(self.omega_w)])
        self.calculated = False
        return omega_w

    def be_verbose(self, verbose=True):
        """Provide verbose messaging on rank 0"""

        self.verbose = verbose
        return verbose

    def verboseprint(self, *args, **kwargs):
        """MPI-safe verbose print
        Prints only from master if verbose is True."""

        if self.verbose:
            parprint(*args, **kwargs)

    def use_singlet(self, singlet=True):
        """Perform as a singlet calculation
        This requires any parallelization to not be over spin"""

        self.singlet = singlet
        return singlet

    def use_hilbert_transform(self, hilbert_transform=True):
        """Use Hilbert Transform?
        hilbert_transform	True/False"""

        if hilbert_transform:
            self.verboseprint('Using Hilbert Transform')
        self.hilbert_transform = hilbert_transform
        return hilbert_transform

    def write_dielectric_function(self, outfilename):
        """Write both real and imaginary part of the dielectric function
        and transitions if calculated"""

        self.calculate()
        if world.rank is 0:
            self.verboseprint('Writing Real Part of Dielectric Function')
            self.__write_function(self.re_epsilon_qvw, outfilename+'_Re_epsilon')
            self.verboseprint('Writing Imaginary Part of Dielectric Function')
            self.__write_function(self.im_epsilon_qvw, outfilename+'_Im_epsilon')
            if self.calculate_transitions():
                self.verboseprint('Writing Transitions')
                self.write_transitions(self.transitionslist, outfilename)
        return

    def write_polarizability(self, outfilename, dim='3D', axis='z'):
        """Write both real and imaginary part of the polarizability
        outfilename	File name of output with
        		'_Re_sigma.dat' and '_Im_sigma.dat' appended
        dim		Axis for determining polarizability prefactor"""

        alpha = self.get_alpha(dim=dim, axis=axis)
        if world.rank is 0:
            self.verboseprint('Writing Real Part of Polarizability')
            self.__write_function(alpha[1], outfilename+'_Re_alpha')
            self.verboseprint('Writing Imaginary Part of Polarizability')
            self.__write_function(alpha[2], outfilename+'_Im_alpha')
        return

    def write_optical_conductivity(self, outfilename, dim='2D', axis='z'):
        """Write both real and imaginary part of the optical conductivity
        outfilename	File name of output with
        		'_Re_sigma.dat' and '_Im_sigma.dat' appended
        dim		Axis for determining conductivity prefactor"""

        sigma = self.get_sigma(dim=dim, axis=axis)
        if world.rank is 0:
            self.verboseprint('Writing Real Part of Optical Conductivity')
            self.__write_function(sigma[1], outfilename+'_Re_sigma')
            self.verboseprint('Writing Imaginary Part of Optical Conductivity')
            self.__write_function(sigma[2], outfilename+'_Im_sigma')
        return

    def write_transitions(self, transitionslist, outfilename):
        """Write the transitions list to a file, sorted by energy in eV"""

        outfilename = outfilename + '_cuttrans'+str(self.cuttrans)+'.dat'
        filehandle = open(outfilename, 'w')
        axes = {0: 'x', 1: 'y', 2: 'z'}
        # Order the list by energy
        transitionslist = sorted(transitionslist,
                                 key=lambda energy: energy[0])
        print('# DeltaE (eV)', 'Intensity (a.u.)', ' i -> j',
              'spin', 'k-point', 'direction', file=filehandle)
        for transition in transitionslist:
            print(transition[:6], axes[transition[6]], file=filehandle)
        return

    def __write_function(self, function_qvw, outfilename):
        """Write function(omega) to a file in eV

        function_qvw	function of direction qv and omega
        outfilename	output filename appended with '.dat'"""

        outfilename = outfilename + '.dat'
        filehandle = open(outfilename, 'w')
        omega_w = self.omega_w * HA
        for i, omega in enumerate(omega_w):
            print(omega, function_qvw[:, i].sum(),
                  function_qvw[0, i], function_qvw[1, i], function_qvw[2, i],
                  file=filehandle)
        filehandle.close()
        return

    def get_grad_phi(self):
        """Calculate grad_phi_kqvnumu matrix of LCAO orbitals"""
        ksl = self.calc.wfs.ksl
        if gpaw_version >= '1.5.0':
            gcomm = self.calc.wfs.gd.comm
            manytci = self.calc.wfs.manytci
            grad_phi_kqvnumu = manytci.O_qMM_T_qMM(gcomm,
                                                   ksl.Mstart,
                                                   ksl.Mstop,
                                                   ignore_upper=ksl.using_blacs,
                                                   derivative=True)[0]
        else:
            spos_ac = self.calc.get_atoms().get_scaled_positions() % 1.0
            nkpts = len(self.calc.wfs.kd.ibzk_qc)
            dtype = self.calc.wfs.dtype
            grad_phi_kqvnumu = empty((nkpts, 3, ksl.mynao, ksl.nao), dtype)
            dtdr_kqvnumu = empty((nkpts, 3, ksl.mynao, ksl.nao), dtype)
            dprojdr_aqvnui = {}
            for natom in self.calc.wfs.basis_functions.my_atom_indices:
                i = self.calc.wfs.setups[natom].ni
                dprojdr_aqvnui[natom] = empty((nkpts, 3, ksl.nao, i), dtype)
            self.calc.wfs.tci.calculate_derivative(spos_ac,
                                                   grad_phi_kqvnumu,
                                                   dtdr_kqvnumu,
                                                   dprojdr_aqvnui)

        # If using BLACS initialize lower triangle.
        if self.calc.wfs.ksl.using_blacs:
            for i in range(grad_phi_kqvnumu.shape[-1]):
                grad_phi_kqvnumu[:, :, i, i:] = -grad_phi_kqvnumu[:, :, i:, i]

        return grad_phi_kqvnumu

    def get_proj_ani(self, spin=0, k=0):
        """Obtain Projector Matrix"""

        # We can work in the IBZ since we have no momentum transfer
        nks = self.calc.wfs.kd.get_rank_and_index(s=spin, k=k)[1]
        kpt = self.calc.wfs.kpt_u[nks]
        nsym = self.calc.wfs.kd.sym_k[k]
        proj_ani = {}
        for a_sa in kpt.P_ani:
            proj_ani[a_sa] = dot(kpt.P_ani[a_sa],
                                 self.calc.wfs.setups[a_sa].R_sii[nsym])
        return proj_ani

    def get_paw_overlap(self, spin1=0, spin2=0, k=0):
        """Calculate PAW Corrections to the overlap qvnm matrix"""

        nbands = self.calc.get_number_of_bands()
        dtype = self.calc.wfs.dtype
        paw_overlap_qvnm = zeros((3, nbands, nbands), dtype=dtype)
        proj_ani = self.get_proj_ani(spin1, k)
        # PAW corrections from other spin channel
        proj_ami = self.get_proj_ani(spin2, k)
        nocc = self.nocc
        setups = self.calc.wfs.setups
        for qdir in range(3):
            for nat in proj_ani:
                # Take PAW corrections for n >= nocc
                # from other spin channel if different
                if spin1 != spin2:
                    proj_ani[nat][nocc:, :] = proj_ami[nat][nocc:, :]
                paw_overlap_qvnm[qdir, :, :] += dot(
                    proj_ani[nat],
                    dot(setups[nat].nabla_iiv[:, :, qdir],
                        proj_ani[nat].transpose()))
        return paw_overlap_qvnm

    def get_df_nm(self, occupations_n, weight, number_of_spins):
        """Apply occupation cutoff cutocc to df_nmot-line

        occupations_n	Array of occupations
        weight			k-point weight
        number_of_spins	numer of spins"""

        # Weighted occupation cutoff
        cutocc = self.cutocc * weight
        # full occupation number
        fullocc = 2. / number_of_spins * weight
        nbands = len(occupations_n)
        # occupancy difference matrix
        df_nm = ((outer(occupations_n, ones(nbands)) -
                  outer(ones(nbands), occupations_n)))
        # Apply cutoff for occupations to remove
        # electronic temperature effects
        if cutocc:
            df_nm[:, :] = ((abs(df_nm[:, :]) > cutocc) * df_nm[:, :])
            df_nm[:, :] += (
                (df_nm[:, :] > fullocc - cutocc) *
                (fullocc - df_nm[:, :]))
            df_nm[:, :] += (
                (df_nm[:, :] < cutocc - fullocc) *
                (-fullocc - df_nm[:, :]))
        return df_nm

    def swap_unoccupied(self, eigenvalues_n, occupations_n, coeff_nnu, mynks):
        """Perform a pseudo-singlet calculation based on a
        triplet calculation by swapping spin channels for n >= nocc
        Takes eigenvalues, occupations, and LCAO coefficients
        for n >= nocc from other spin channel

        eigenvalues_n		Array of eigenvalues
        occupations_n		Array of occupations
        coeff_nnu		Matrix of LCAO coefficients

        spin			spin of other spin channel"""

        kdesc = self.calc.wfs.kd  # k-point descriptor
        spin, k = kdesc.what_is(kdesc.global_index(mynks))
        # other spin channel
        spin = (spin + 1) % self.calc.get_number_of_spins()
        # The singlet mode does not support spin parallelization
        assert kdesc.get_rank_and_index(s=spin, k=k)[0] == self.calc.comms['k'].rank
        nks = kdesc.where_is(s=spin, k=k)
        kpt_u = self.calc.wfs.kpt_u[nks]
        # Take eigenvalues, occupations, and LCAO coefficients
        # for n >= nocc from other spin channel
        nocc = self.nocc
        eigenvalues_n[nocc:] = kpt_u.eps_n[nocc:]
        occupations_n[nocc:] = kpt_u.f_n[nocc:]
        coeff_nnu[nocc:] = kpt_u.C_nM[nocc:]
        return spin

    def get_overlap_energydiff(self, k, spin, mynks):
        """Calculate overlap matrices and energy difference matrix
        for requested k-point and spin with index mynks

        k		global k-point index
        spin		global spin index
        mynks		local k-point and spin index

        overlap_qvnm	overlap matrix of direction and bands
        de_nm		energy difference matrix of bands"""

        nbands = self.calc.get_number_of_bands()    # total number of bands
        identity_nm = identity(nbands, dtype=self.calc.wfs.dtype)  # identity
        eigenvalues_n = self.calc.wfs.kpt_u[mynks].eps_n.copy()  # eigenvalues
        occupations_n = self.calc.wfs.kpt_u[mynks].f_n.copy()  # occupations
        coeff_nnu = self.calc.wfs.kpt_u[mynks].C_nM.copy()  # LCAO coefficients
        
        #Include the GLLBSC scissors correction, if implemented
        eigenvalues_n[self.nocc:] += self.scissor
        
        if self.singlet:
            spin2 = self.swap_unoccupied(eigenvalues_n, occupations_n,
                                         coeff_nnu, mynks)
        else:
            spin2 = spin
        # Calculating PAW Corrections
        de_nm = (outer(eigenvalues_n, ones(nbands)) -
                 outer(ones(nbands), eigenvalues_n))
        df_nm = self.get_df_nm(occupations_n,
                               weight=self.calc.wfs.kd.weight_k[k],
                               number_of_spins=self.calc.get_number_of_spins())
        gradcoeff_num = zeros(coeff_nnu.shape, dtype=self.calc.wfs.dtype)
        # Overlap Matrix initialized with PAW overlaps
        overlap_qvnm = self.get_paw_overlap(spin1=spin,
                                            spin2=spin2, k=k)
        # Calculating Overlap Matrix
        # Perform Matrix Multiplications for each direction
        nkpts = self.grad_phi_kqvnumu.shape[0]
        for qdir in range(3):
            # Be careful that mynks points to the k-point index
            gemm(1.0,
                 self.grad_phi_kqvnumu[mynks % nkpts, qdir],
                 coeff_nnu, 0.0, gradcoeff_num)
            gemm(1.0, coeff_nnu, gradcoeff_num, 1.0, overlap_qvnm[qdir], 'c')
            overlap_qvnm[qdir] /= de_nm + identity_nm
        # Sum overlap_qvnm over domains
        self.calc.comms['K'].sum(overlap_qvnm)
        overlap_qvnm = self.prefactor * df_nm * (
            overlap_qvnm * overlap_qvnm.conj())
        return overlap_qvnm.real, de_nm

    def calculate_transitions(self, do_transitions=None, cuttrans=1e-2):
        """Specify whether to calculate transition intensities
        do_transitions True/False
        Resets calculated to False if do_transitions is True"""

        if do_transitions is not None:
            self.cuttrans = cuttrans
            self.do_transitions = do_transitions
            self.calculated = (not do_transitions) and self.calculated
            if not self.calculated:
                self.transitionslist = []
        return self.do_transitions

    def __update_transitions(self, overlap_qvnm, de_nm, spin, k):
        """Update transition intensities list"""

        nbands = self.calc.get_number_of_bands()
        transitions_qvnm = empty(overlap_qvnm.shape)
        omega_nm = de_nm.copy() * HA
        omegamin = self.omega_w.min()
        omegamax = self.omega_w.max()
        for qdir in range(3):
            if self.hilbert_transform:
                transitions_qvnm[qdir] = self.eta * overlap_qvnm[qdir] * (
                    1 / ((abs(de_nm) - de_nm)**2 + self.eta**2) -
                    1 / ((abs(de_nm) + de_nm)**2 + self.eta**2))
            else:
                transitions_qvnm[qdir] = -overlap_qvnm[qdir] / self.eta
            for i in range(0, nbands-1):
                for j in range(i+1, nbands):
                    if omegamin < abs(de_nm[i, j]) and abs(de_nm[i, j]) < omegamax:
                        if transitions_qvnm[qdir, i, j] > self.cuttrans:
                            self.transitionslist.append([
                                -omega_nm[i, j],
                                transitions_qvnm[qdir, i, j],
                                i, j, spin, k, qdir])
        return

    def __gather_transitions(self):
        """Gather list of transitions onto rank 0"""

        n_indices = 7
        # Employ communicator for different k-points
        kptcomm = self.calc.comms['k']
        if kptcomm.rank is 0:
            for rank in range(1, kptcomm.size):
                ntrans = array(0)
                kptcomm.receive(ntrans, src=rank)
                data = empty([ntrans, n_indices])
                kptcomm.receive(data, src=rank)
                for i in range(ntrans):
                    self.transitionslist.append(data[i].tolist())
        else:
            ntrans = array(len(self.transitionslist))
            kptcomm.send(ntrans, dest=0)
            kptcomm.send(array(self.transitionslist), dest=0)
        return

    def __update_epsilon(self, overlap_qvnm, de_nm):
        """Update real and imaginary parts of the
        dielectric function epsilon using the given
        overlap and energy difference matricies
        overlap_qvnm         Matrix Elements
        de_nm			Energy Differences"""

        if self.hilbert_transform:
            u_indices = triu_indices(overlap_qvnm.shape[-1], 1)
        omega_w = self.omega_w
        re_epsilon_qvw = self.re_epsilon_qvw
        im_epsilon_qvw = self.im_epsilon_qvw
        for qdir in range(3):
            for i, omega in enumerate(omega_w):
                if self.hilbert_transform:
                    re_epsilon_qvw[qdir, i] += (
                        overlap_qvnm[qdir][u_indices] * (
                            (de_nm[u_indices] - omega) /
                            (self.eta**2 +
                             (omega - de_nm[u_indices])**2) -
                            (de_nm[u_indices] - omega) /
                            (self.eta**2 +
                             (omega + de_nm[u_indices])**2)
                        )).sum()
                    im_epsilon_qvw[qdir, i] += self.eta * (
                        overlap_qvnm[qdir][u_indices] * (
                            1 / (self.eta**2 +
                                 (omega - de_nm[u_indices])**2) -
                            1 / (self.eta**2 +
                                 (omega + de_nm[u_indices])**2)
                        )).sum()
                else:
                    re_epsilon_qvw[qdir, i] += (
                        (de_nm - omega) *
                        overlap_qvnm[qdir] / (
                            self.eta**2 + (omega - de_nm)**2
                        )).sum()
                    im_epsilon_qvw[qdir, i] += self.eta * (
                        overlap_qvnm[qdir] / (
                            self.eta**2 + (omega - de_nm)**2
                        )).sum()
        return

    def calculate(self, recalculate=False):
        """Calculate the dielectric function the q->0+ Optical Limit"""

        # Employ communicator for different k-points
        kptcomm = self.calc.comms['k']
        if not self.calculated or recalculate:
            # Loop over k-point and spin on local rank
            for mynks, nks in enumerate(
                    self.calc.wfs.kd.get_indices(rank=kptcomm.rank)):
                spin, k = self.calc.wfs.kd.what_is(u=nks)
                self.verboseprint('Calculating Matrix Element', nks)
                overlap_qvnm, de_nm = self.get_overlap_energydiff(
                    k=k, spin=spin, mynks=mynks)
                if self.calculate_transitions():
                    self.verboseprint('Calculating Transitions')
                    self.__update_transitions(overlap_qvnm, de_nm, spin, k)
                # Calculating Optical Absorption
                if self.hilbert_transform:
                    self.verboseprint('Using Hilbert Transform')
                self.__update_epsilon(overlap_qvnm, de_nm)
            # Gather and sum real and imaginary epsilon
            # over k-points and send to master node
            self.verboseprint('Summing Contributions')
            kptcomm.sum(self.re_epsilon_qvw)
            kptcomm.sum(self.im_epsilon_qvw)
            if self.calculate_transitions():
                # Gather transitions on the master node
                self.__gather_transitions()
            # Allows the calculate function to be called multiple times
            self.calculated = True
        return

    def get_epsilon(self):
        """Returns arrays with the
        real and imaginary parts of the
        dielectric function in each direction
        and the energy range in eV

        omega_w		Energy range of spectra in eV
        re_epsilon_qvw	Real part of epsilon
        im_epsilon_qvw	Imaginary part of epsilon"""

        self.calculate()  # Ensure results are calculated
        return (self.omega_w * HA, self.re_epsilon_qvw,
                self.im_epsilon_qvw)

    def get_alpha(self, dim='3D', axis='z'):
        """Returns arrays with the
        real and imaginary parts of the
        0D, 1D, 2D, or 3D polarizability in each direction
        and the energy range in eV using
        alpha_par = prefactor*(epsilon(omega) - 1)
        alpha_perp = prefactor*(1 - 1/epsilon(omega))
        dim   	Dimensionality of the system for deteriming prefactor
            '0D' or 0, prefactor = volume/(4*pi)
            Default: volume = abs(dot(cell[0], cross(cell[1], cell[2])))
    		'1D' or 1, prefactor = perpendicularcellarea/(4*pi)
    		Default: perpendicularcellarea = cross(cell[0], cell[1])
    		'2D' or 2, prefactor = normalcelllength/(4*pi)
        	Default: normalcelllength = cell[2,2]
        axis	Preferential axis (Default axis='z')
        	'x', 'y', or 'z' if string
		0, 1, 2 if integer
		1D: axis of one dimensional structure
        2D: normal to two dimensional structure

        omega_w		Energy range of spectra in eV
    	re_alpha_qvw	Real part of alpha
    	im_alpha_qvw	Imaginary part of alpha"""
        self.get_epsilon()
        # Parse axis
        if isinstance(axis, str):
            axes = {'x': 0, 'y': 1, 'z': 2}
            if axis not in axes.keys():
                raise NotImplementedError('Invalid axis', axis)
            axis = axes[axis]
        # Parse dimensionality
        if isinstance(dim, int):
            dim = str(int)+'D'
        prefactor = 1 / (4 * pi)
        cell = self.calc.wfs.gd.cell_cv
        if dim == '0D':
            perp = [0, 1, 2]
            # prefactor includes the entire volume of the unit cell
            prefactor *= abs(dot(cell[0], cross(cell[1], cell[2])))       
        elif dim == '1D':
            perp = [mod(axis - 1, 3), mod(axis + 1, 3)]
            # prefactor includes unit cell area normal to axis
            prefactor *= cross(cell[perp[0]], cell[perp[1]])
        elif dim == '2D':
            perp = [axis]
            # prefactor includes unit cell length normal to plane
            prefactor *= cell[axis, axis]
        elif dim == '3D':
            perp = []
            # prefactor is just 1 / (4 * pi)
        else:
            raise NotImplementedError('Dimensionality', dim)
        # Calculate alpha parallel to structure
        re_alpha_qvw = (self.re_epsilon_qvw.copy() - 1)
        im_alpha_qvw = self.im_epsilon_qvw.copy()
        # Calculate alpha perpendicular to structure
        for i in perp:
            re_alpha_qvw[i] = ((self.re_epsilon_qvw[i] - (self.re_epsilon_qvw[i]**2 +
                                                          self.im_epsilon_qvw[i]**2)) /
                               (self.re_epsilon_qvw[i]**2 + self.im_epsilon_qvw[i]**2))
            im_alpha_qvw[i] = (self.im_epsilon_qvw[i] /
                               (self.re_epsilon_qvw[i]**2 + self.im_epsilon_qvw[i]**2))
        # Include prefactor
        re_alpha_qvw *= prefactor
        im_alpha_qvw *= prefactor
        return (self.omega_w * HA, re_alpha_qvw, im_alpha_qvw)

    def get_sigma(self, dim='2D', axis='z'):
        """Returns arrays with the
        real and imaginary parts of the
        optical conductivity in each direction
        and the energy range in eV using
        sigma = - i * omega * alpha(omega)
        dim   	Dimensionality of the system for deteriming prefactor
    		'1D' or 1, prefactor = perpendicularcellarea/(4*pi)
    		Default: perpendicularcellarea = cross(cell[0], cell[1])
    		'2D' or 2, prefactor = normalcelllength/(4*pi)
        	Default: normalcelllength = cell[2,2]
        axis	Preferential axis (Default axis='z')
        	'x', 'y', or 'z' if string
		0, 1, 2 if integer
		1D: axis of one dimensional structure
        	2D: normal to two dimensional structure

        omega_w	 	Energy range of spectra in eV
        re_sigma_qvw	Real part of sigma
        im_sigma_qvv	Imaginary part of sigma"""
        omega_w, im_sigma_qvw, re_sigma_qvw = self.get_alpha(dim, axis)
        re_sigma_qvw *= self.omega_w
        im_sigma_qvw *= -self.omega_w
        return (omega_w, re_sigma_qvw, im_sigma_qvw)

def read_arguments():
    """Input Argument Parsing"""

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='name of input GPAW file')
    parser.add_argument('-o', '--outfilename',
                        help='output file (default: filename)',
                        type=str, default='')
    parser.add_argument('-v', '--verbose',
                        help='increase output verbosity',
                        action='store_true')
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
    parser.add_argument('-c', '--cutocc',
                        help='cutoff for |f_n - f_m| (%(default)s)',
                        default=1e-5, type=float)
    parser.add_argument('-ht', '--hilbert_transform',
                        help='use Hilbert transform (%(default)s)',
                        action='store_true')
    parser.add_argument('-s', '--singlet',
                        help='s=0 -> s=1 and s=1 -> s=0 (%(default)s)',
                        action='store_true')
    parser.add_argument('-df', '--epsilon',
                        help='output dielectric function (%(default)s)',
                        action='store_false')
    parser.add_argument('-t', '--transitions',
                        help='output optical transitions (%(default)s)',
                        action='store_true')
    parser.add_argument('-pol', '--alpha',
                        help='output polarizability (%(default)s)',
                        action='store_true')
    parser.add_argument('-oc', '--sigma',
                        help='output optical conductivity (%(default)s)',
                        action='store_true')
    parser.add_argument('-sc', '--scalapack',
                        help='use ScaLAPACK parallelization (%(default)s)',
                        action='store_true')
    parser.add_argument('-d', '--dim',
                        help='Dimensionality of the system (%(default)s)',
                        default='3D', type=str)
    parser.add_argument('-ax', '--axis',
                        help='Axis parallel (1D) or perpendicular (2D) to the system (%(default)s)',
                        default='z', type=str)
    parser.add_argument('-ct', '--cuttrans',
                        help='cutoff for transitions (%(default)s)',
                        default=1e-2, type=float)
    parser.add_argument('-ddc', '--scissor',
                        help='GLLB-SC correction for unoccupied band eigenvalues in eV (%(default)s)',
                        default=0, type=float)
    pargs = parser.parse_args()
    if pargs.outfilename == '':
        outfilename = pargs.filename.split('.gpw')[0]
    else:
        outfilename = pargs.outfilename
    outfilename += '_cut' + str(pargs.cutocc)
    outfilename += '_singlet' if pargs.singlet else ''
    outfilename += '_HT' if pargs.hilbert_transform else ''
    pargs.outfilename = outfilename
    return pargs

def main():
    """Command Line Executable"""
    # Read Arguments
    args = read_arguments()
    # Initialize LCAOTDDFTq0 object
    tddft = LCAOTDDFTq0(args.filename,
                        eta=args.eta,
                        scissor=args.scissor,
                        cutocc=args.cutocc,
                        verbose=args.verbose)
    tddft.set_energy_range(omegamin=args.omegamin,
                           omegamax=args.omegamax,
                           domega=args.domega)
    tddft.use_hilbert_transform(args.hilbert_transform)
    tddft.use_singlet(args.singlet)
    tddft.calculate_transitions(args.transitions,
                                cuttrans=args.cuttrans)
    # Calculate and output dielectric function and transitions
    if args.epsilon:
        tddft.write_dielectric_function(args.outfilename)
    # Calculate and output polarizability
    if args.alpha:
        tddft.write_polarizability(args.outfilename,
                                   args.dim,
                                   args.axis)
    # Calculate and output optical conductivity
    if args.sigma:
        tddft.write_optical_conductivity(args.outfilename,
                                         args.dim,
                                         args.axis)

if __name__ == '__main__':
    main()

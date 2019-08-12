import os
import re
import numpy as np
import sys
import subprocess
from os.path import exists

import matplotlib.tri as tri
import matplotlib.colors as col
from matplotlib import pyplot as plt
import pylab as p

"""
In order to create 1D projection plots, we first need cube files at each of the energies of interest. For example, within SLURM, the following provides an efficient way of executing the exciton_density.py file for energies between 0.1-20.0eV with a 0.1eV spacing (notice the array goes from 1-200 and is divided by 10 before being fed to the code):

#SBATCH --array=1-200

calc(){ awk "BEGIN { print "$*" }"; }
echo $(calc $SLURM_ARRAY_TASK_ID/10) | (read energy; mpirun -np 8 gpaw-python exciton_density.py example.gpw example_transitions.dat -w=$energy;)

Afterwards, since all three possible polarization directions are output by the exciton_density.py code, partition off the cube files corresponding to the desired polarization direction (either x, y, or z).
"""

def read_cube_data(cube_file):
    #Reads the data from the cube file into a more useful form for plotting

    at_coord=[]
    spacing_vec=[]
    nline=0
    values=[]
     # Read cube file and parse all data
    for line in open(cube_file, "r"):
        nline+=1
        if nline==3:
            try:
                nat=int(line.split()[0]) 
                origin=[float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
            except:
                print("ERROR: non recognized cube format")
        elif nline >3 and nline <= 6:
            spacing_vec.append(line.split())
        elif nline > 6 and nline <= 6+nat:
            at_coord.append(line.split())
        elif nline > 5:
            if nline > 6+nat:
                for i in line.split():
                    values.append(float(i))

    return spacing_vec, values


def number_voxels(cube_file):
    # Gives the number of voxels in each direction, useful when using noproj

    spacing_vec, values = read_cube_data(cube_file)
    print('x: '+str(spacing_vec[0][0])+'\n'+'y: '+str(spacing_vec[1][0])+'\n'+'z: '+str(spacing_vec[2][0]))


def projection_data_z(cube_file, cube_projection):
    # This method yields cube data projected along the z-axis

    spacing_vec, values = read_cube_data(cube_file)
    idx=-1
    length = int(spacing_vec[2][0])*float(spacing_vec[2][3])*0.5291772083
    # This yields the length of unit cell in Angstrom

    if not exists(cube_projection):
        data = np.zeros(shape=(int(spacing_vec[2][0])+1, 2))

        for i in range(0, int(spacing_vec[0][0])):
            for j in range(0, int(spacing_vec[1][0])):
                for k in range(0, int(spacing_vec[2][0])):
                    idx+=1
                    # This translates the values of the cube file to centre on the zero of the axis
                    data[k, 0] = (k-int(spacing_vec[2][0])/2.0)*(length/(int(spacing_vec[2][0])))
                    data[k, 1] += values[idx]
        z = (k+1-int(spacing_vec[2][0])/2.0)*(length/(int(spacing_vec[2][0])))
        data[k+1, 0] = z
        data[k+1, 1] = data[0][1] # Required since the cube data does not include periodic replications       

        data = sorted(data, key=lambda x: x[0])

        # Prints the output to a .dat file so it can easily be read in for future contour plots
        tmp=sys.stdout
        sys.stdout = open(cube_projection, 'w')
        for i in range(0, len(data)):
            print(data[i][0], data[i][1])
        sys.stdout.close()
        sys.stdout=tmp
        print("Printing file "+str(cube_projection))
    else:
        lines = open(cube_projection, 'r').readlines()
        data = []
        for line in lines:
            data.append(np.fromstring(line, dtype=float, sep=' '))
        data=np.array(data)    

    return data, length


def projection_data_y(cube_file, cube_projection, noproj = False, voxel = '0'):
    # This method yields cube data projected along the y-axis

    spacing_vec, values = read_cube_data(cube_file)
    idx=-1
    length = int(spacing_vec[1][0])*np.sqrt(float(spacing_vec[1][1])**2 + float(spacing_vec[1][2])**2)*0.5291772083
    # This yields the length of unit cell in Angstrom

    if not exists(cube_projection):
        data3D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0]), int(spacing_vec[2][0])))
        data2D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0])))
        data = [] # y-axis projection plot

        for i in range(0, int(spacing_vec[0][0])):
            for j in range(0, int(spacing_vec[1][0])):
                for k in range(0, int(spacing_vec[2][0])):
                    idx+=1
                    data3D[i, j, k]=values[idx]
                data2D[i, j]=data3D[i, j, :].sum() # Sums out z, the outer column

        for j in range(0, int(spacing_vec[1][0])):
            rho_j = 0
            # Centers the cube files on the origin
            y = (j-int(spacing_vec[1][0])/2.0)*(length/(int(spacing_vec[1][0])))
            if noproj:
                rho_j = data2D[int(voxel), j]
            else:
                for i in range(0, int(spacing_vec[0][0])):
                    # This sums along x-axis
                    rho_j+=data2D[i, j] 
            data.append([y, rho_j])
        y = (j+1-int(spacing_vec[1][0])/2.0)*(length/(int(spacing_vec[1][0])))
        data.append([y, data[0][1]]) # Required since the cube data does not include periodic replications 
         
        data = sorted(data, key=lambda x: x[0])

        # Prints the output to a .dat file so it can easily be read in for future contour plots
        tmp=sys.stdout
        sys.stdout = open(cube_projection,'w')
        for i in range(0, len(data)):
            print(data[i][0], data[i][1])
        sys.stdout.close()
        sys.stdout=tmp
        print("Printing file "+str(cube_projection))
    else:
        lines = open(cube_projection, 'r').readlines()
        data = []
        for line in lines:
            data.append(np.fromstring(line, dtype=float, sep=' '))
        data=np.array(data)

    return data, length


def projection_data_x(cube_file, cube_projection, noproj = False, voxel = '0'):
    # This method yields cube data projected along the x-axis

    spacing_vec, values = read_cube_data(cube_file)
    idx=-1
    length = int(spacing_vec[0][0])*np.sqrt(float(spacing_vec[0][1])**2 + float(spacing_vec[0][2])**2)*0.5291772083
    # This yields the length of unit cell in Angstrom

    if not exists(cube_projection):
        data3D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0]), int(spacing_vec[2][0])))
        data2D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0])))
        data = [] # x-axis projection plot

        for i in range(0, int(spacing_vec[0][0])):
            for j in range(0, int(spacing_vec[1][0])):
                for k in range(0, int(spacing_vec[2][0])):
                    idx+=1
                    data3D[i, j, k]=values[idx]
                data2D[i, j]=data3D[i, j, :].sum() # Sums out z, the outer column

        for i in range(0, int(spacing_vec[0][0])):
            rho_i = 0
            # Centers the cube files on the origin
            x = (i-(int(spacing_vec[0][0]))/2.0)*(length/(int(spacing_vec[0][0])))
            if noproj:
                rho_i = data2D[i, int(voxel)]
            else:                 
                for j in range(0, int(spacing_vec[1][0])):      
                    # This sums along y-axis
                    rho_i+=data2D[i, j]
            data.append([x, rho_i])
        x = (i+1-(int(spacing_vec[0][0]))/2.0)*(length/(int(spacing_vec[0][0])))
        data.append([x, data[0][1]]) # Required since the cube data does not include periodic replications 
         
        data = sorted(data, key=lambda x: x[0])

        # Prints the output to a .dat file so it can easily be read in for future contour plots
        tmp=sys.stdout
        sys.stdout = open(cube_projection, 'w')
        for i in range(0, len(data)):
            print(data[i][0], data[i][1])
        sys.stdout.close()
        sys.stdout=tmp
        print("Printing file "+str(cube_projection))
    else:
        lines = open(cube_projection, 'r').readlines()
        data = []
        for line in lines:
            data.append(np.fromstring(line, dtype=float, sep=' '))
        data=np.array(data)

    return data, length


def projection_data_diagonal(cube_file, cube_projection, diag_dir = 'nwse', angle = '90'):
    # This method yields cube data projected along the "diagonal axis"
    # This refers to z-axis data being summed out, and 

    # diag_dir: 'nwse' goes from (N, 0) voxel to (0, N) voxel; 'nesw' goes from (0, 0) voxel to (N, N) voxel
    # angle: angle between the two axes, useful for non-orthorhombic cells (degrees)

    angle = np.pi/180.0*float(angle) # Change to radians

    spacing_vec, values = read_cube_data(cube_file)
    idx=-1
    length1 = int(spacing_vec[0][0])*np.sqrt(float(spacing_vec[0][1])**2 + float(spacing_vec[0][2])**2)*0.5291772083
    length2 = int(spacing_vec[1][0])*np.sqrt(float(spacing_vec[1][1])**2 + float(spacing_vec[1][2])**2)*0.5291772083
    length = np.sqrt(length1**2 + length2**2 - 2*length1*length2*np.cos(angle))
    # This yields the length of the bond-axis in Angstrom. Change the angle here depending on shape on unit cell

    if not exists(cube_projection):
        data3D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0]), int(spacing_vec[2][0])))
        data2D = np.zeros(shape=(int(spacing_vec[0][0]), int(spacing_vec[1][0])))
        data = [] # x-axis projection plot

        for i in range(0, int(spacing_vec[0][0])):
            for j in range(0, int(spacing_vec[1][0])):
                for k in range(0, int(spacing_vec[2][0])):
                    idx+=1
                    data3D[i, j, k]=values[idx]
                data2D[i, j]=data3D[i, j, :].sum() # Sums out z, the outer column
        Lx = int(spacing_vec[0][0])

        for i in range(0, Lx+1):
            if diag_dir == 'nwse':
                rho_i=data2D[(i)%Lx,(Lx-i)%Lx] 
                data.append([(i-Lx/2)*length/Lx, rho_i])
            if diag_dir == 'nesw':
                rho_i=data2D[(i)%Lx,(i)%Lx] 
                data.append([(i-Lx/2)*length/Lx, rho_i])                
         
        data = sorted(data, key=lambda x: x[0])

        # Prints the output to a .dat file so it can easily be read in for future contour plots
        tmp=sys.stdout
        sys.stdout = open(cube_projection, 'w')
        for i in range(0, len(data)):
            print(data[i][0], data[i][1])
        sys.stdout.close()
        sys.stdout=tmp
        print("Printing file "+str(cube_projection))
    else:
        lines = open(cube_projection, 'r').readlines()
        data = []
        for line in lines:
            data.append(np.fromstring(line, dtype=float, sep=' '))
        data=np.array(data)

    return data, length


def contour_maker(dtype = 'e', direction = 'z', cubedir='.', coarseness=1, brightness=0.75, emin=0.2, emax=20, ax=p, xenergy=False, noproj=False, voxel = '0', diag_dir = 'nwse', angle = '90'):
    # This creates 1D projection plots along either the x, y, or z axes when given energy-resolved cube file data

    # type: 'e' or 'h', depending on the type of density you want plotted
    # direction: 'x', 'y', 'z', or 'diag' will project all cube directions onto the chosen axis
    # cubedir: the location of the cube files
    # coarseness: Choose the resolution of the energy; a higher coarseness will result in a larger energy spacing
    # brightness: Choose the brightness of the plot
    # emin: the minimum energy plotted on the contour plot's x-axis
    # emax: the maximum energy plotted on the contour plot's x-axis
    # ax: the frame in matplotlib, change from default if subplots are desired
    # xenergy: multiply the values of the contour plot by the energy to better resolve the spectra at higher energies
    # noproj: instead of doing a projection plot, choose to see a slice of 2D projected data at each energy
    # voxel: choose the voxel along the other direction at which the 'noproj' slice is taken. Use number_voxels() for help
    # diag_dir: 'nwse' goes from (N, 0) voxel to (0, N) voxel; 'nesw' goes from (0, 0) voxel to (N, N) voxel
    # angle: angle between the two axes, useful for non-orthorhombic cells when using projection_data_diagonal (degrees)

    fnames = os.listdir(cubedir)
    codedir = os.getcwd()

    if dtype == 'e':
        postpend = 'e.cube'
    if dtype == 'h':
        postpend = 'h.cube'

    Axis = [] # y-axis of contour plot: graphene z-axis
    Density = [] # colours: electron densities
    energies  = [] # x-axis of contour plot: energies

    os.chdir(cubedir)
    for f in fnames:
        if f.find(postpend) != -1:
            energy = re.findall(r'[-+]?(\d*\.\d*)', f)[0]
            if np.ceil(float(energy)*100)%int(coarseness) == 0: 
                energies.append(energy)
                # The z-axis method only does summation, so noproj = False for these calculations
                if direction == 'z':
                    data, length = projection_data_z(f, 'z_{}eV_'.format(energy)+dtype+'.dat')
                # The y-axis method can do 1D projection or 2D slices, depending on the value of noproj
                if direction == 'y':
                    if noproj:
                        data, length = projection_data_y(f, 'y_{}eV_'.format(energy)+'voxel_'+str(voxel)+'_'+dtype+'.dat', noproj, voxel)
                    else:
                        data, length = projection_data_y(f, 'y_{}eV_'.format(energy)+dtype+'.dat')
                # The x-axis method can do 1D projection or 2D slices, depending on the value of noproj
                if direction == 'x':
                    if noproj:
                        data, length = projection_data_x(f, 'x_{}eV_'.format(energy)+'voxel_'+str(voxel)+'_'+dtype+'.dat', noproj, voxel)
                    else:
                        data, length = projection_data_x(f, 'x_{}eV_'.format(energy)+dtype+'.dat')
                # The bond method does not do a summation, so noproj = True for these calculations
                if direction == 'diag':
                    data, length = projection_data_diagonal(f, 'diag_{}eV_'.format(energy)+dtype+'.dat', diag_dir, angle)
                axis = []
                density = []
                for n in range(0, len(data)):
                    axis.append(float(data[n][0]))
                    if dtype == 'e':
                        density.append(-float(data[n][1])*(float(energy) if xenergy else 1.0))
                    if dtype == 'h':
                        density.append(float(data[n][1])*(float(energy) if xenergy else 1.0))
                Axis.append(axis)
                Density.append(density)
    os.chdir(codedir)

    Axis = np.array(Axis)
    Density = np.array(Density)
    energies = np.array(energies)
    energies_w = np.zeros(Axis.shape)
    for i in range(energies_w.shape[1]):
        energies_w[:, i] = energies[:]


    cdict = {'red':  [(0.00, 0.00, 0.00),
                      (0.05, 0.50, 0.50),
                      (0.20, 1.00, 1.00),
                      (1.00, 1.00, 1.00),
                      ],
             'green':[(0.00, 0.00, 0.00),
                      (0.20, 0.00, 0.00),
                      (0.70, 1.00, 1.00),
                      (1.00, 1.00, 1.00),
                      ],
             'blue': [(0.00, 0.00, 0.00),
                      (0.05, 0.50, 0.50),
                      (0.20, 0.00, 0.00),
                      (0.70, 0.00, 0.00),
                      (1.00, 1.00, 1.00),
                      ],
             }

    cmap = col.LinearSegmentedColormap('STM_Colormap', cdict)
    triang = tri.Triangulation(energies_w.flatten(), Axis.flatten())

    Density /= Density.max()
    Density_max = Density.max()/float(brightness)
    Dloss = Density.max()/800/float(brightness)
    zmin = -length/2
    zmax = length/2

    ax.tricontourf(triang, Density.flatten(),
                  np.arange(0, Density_max, Dloss),
                  cmap=cmap, extend='both')

    ax.plot([zmin, zmax], [631, 631], linestyle='--', color='w', linewidth=3.)
    ax.plot([zmin, zmax], [346, 346], linestyle='--', color='w', linewidth=3.)
    ax.plot([zmin, zmax], [296, 296], linestyle='--', color='w', linewidth=3.)
    ax.plot([zmin, zmax], [639, 639], linestyle='-.', color='w', linewidth=3.)
    ax.plot([zmin, zmax], [382, 382], linestyle='-.', color='w', linewidth=3.)
    ax.plot([zmin, zmax], [319, 319], linestyle='-.', color='w', linewidth=3.)
    ax.axis((float(emin), float(emax), zmin, zmax))

    ax.show()
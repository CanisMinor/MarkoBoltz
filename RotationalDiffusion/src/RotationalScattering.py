
'''
MONTE CARLO SIMULATION FOR DISCRETE ROTATIONAL HOPPING
# 14th July 2008

# This program is written for discrete rotational hopping on lattice surfaces
# of single particles and molecules.  The single particles are assumed anchored
# to the centre of their rotational hopping.  The molecules are assumed to
# rotate around their centre of mass.  The scan directions (denoted A and B)
# are defined depending on the lattice symmetry. For square lattices (two-fold
# and four-fold) scan directions are <1 0 0> and <1 1 0>.  For triangular
# lattices (three-fold, six-fold), scan directions are <1 0 -1> and
# <1 1 -2>. (Please see illustrations.)

# First parameter values are defined.  Secondly dependent parameters are
# calculated.  The program then generates a rotatory random walk trajectory.
# A scattering calculation (i.e. calculating ISF) is performed, first for the
# A-direction and then for the B-direction.  For both A and B the scan is
# simulated as a loop over NoK momentum transfer values between 0 and 5
# inverse Angstroms.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as spopt
import Utils as util



def plot_results(momentum, decay, EISF, QEA, slice, phi):
    #plot signature
    plt.plot(momentum,decay,'b') #,momentum,decay,'r+')  #lines and points
    if slice==2 or slice ==4:
        plt.title( 'Signature for phi = ' + str(phi) )
    elif slice == 3 or slice == 6:
        plt.title( 'Signature along phi = ' + str(phi))
    else:
        print( 'Choose a different lattice symmetry.' )
    plt.xlabel( 'Momentum Transfer DeltaK' )
    plt.ylabel( 'Decay Constant of ISF' )
    plt.axis([0, 5, -0.01, 1.01])
    plt.show()


    #plot EISF and QEA
    plt.plot(momentum, EISF, 'b') #,K1p,C1p,'r+')
    plt.plot(momentum, QEA, 'g') #,K1p,D1p,'r+')
    if slice==2 or slice ==4:
        plt.title( 'EISF and QEA Plots for phi = ' + str(phi))
    elif slice == 3 or slice == 6:
        plt.title('EISF and QEA Plots for phi = ' + str(phi))
    else:
        print( 'Choose a different lattice symmetry.' )

    plt.xlabel( 'Momentum Transfer DeltaK' )
    plt.ylabel( 'EISF (blue), QEA (green) ' )
    plt.axis([0, 5, -0.01, 1.01])
    plt.show()

    return

def continuous_trajectory(N_time, dt, N_atoms, N_pots, V0, eta, mass, radius, temperature):
    #initialise convenience parameters
    I = mass * radius * radius   # Moment of inertia
    q0 = 2.0 * eta * mass * temperature * dt
    deltaThetaAtom = (2.0 * math.pi) / float(N_atoms)   #angular separation of atoms

    #initialise angle and speed coordinates
    theta = np.zeros(N_time)
    omega = np.zeros(N_time)

    ## Generate Trajectory
    # Calculate trajectory by fourth-order Runge-Kutta integration
    for k in range(0, N_time):
        c_1 = (-0.5 * N_pots * V0 * math.cos(N_pots * theta[k-1]) - eta * omega[k-1] + np.random.normal() * q0)/I
        c_2 = (-0.5 * N_pots * V0 * math.cos(N_pots * (theta[k-1] + (omega[k-1] + 0.5 * dt * c_1) * dt/2.0)) - eta*(omega[k-1] + 0.5 * dt * c_1) + np.random.normal() * q0)/I
        c_3 = (-0.5 * N_pots * V0 * math.cos(N_pots * (theta[k-1] + (omega[k-1] + 0.5 * dt * c_2) * dt/2.0)) - eta*(omega[k-1] + 0.5 * dt * c_2) + np.random.normal() * q0)/I
        c_4 = (-0.5 * N_pots * V0 * math.cos(N_pots * (theta[k-1] + (omega[k-1]+ dt * c_3) * dt)) - eta*(omega[k-1] + dt*c_3) + np.random.normal() * q0)/I
        omega[k] = omega[k-1] + (c_1 + 2.0 * c_2 + 2.0 * c_3 + c_4) * dt/6.0
        theta[k] = theta[k-1] + omega[k] * dt

    # Convert to Cartesian coordinates
    Rx = np.zeros((N_atoms, N_time))
    Ry = np.zeros((N_atoms, N_time))

    # Convert trajectory to Cartesian coordinates
    for w in range(0, N_atoms):
        Rx[w, :], Ry[w, :] = util.pol2cart(theta + (float(w) * deltaThetaAtom), radius)

    return Rx, Ry

def discrete_trajectory(N_time, N_atoms, pBoltz, slice, radius):
    '''
    function discrete_trajectory
    ---------------------------
    generate random walk hopping along circle
    :param N_time: number of hops
    :param N_atoms: number of atoms hopping
    :param pBoltz: hopping (Boltzmann) probability
    :param slice: number of sites along circle
    :param radius: radius of circle
    :return: Cartesian coordinates Rx, Ry of each atom (rows) in each time step (column)
    '''

    #find angle increment
    deltaTheta = (2.0 * math.pi) / float(slice)        #angular increment between sites on circle
    deltaThetaAtom = (2.0 * math.pi) / float(N_atoms)  #angular separation of atoms in molecule


    #generate vectors P and Q with random numbers between 0 and 1.
    P = np.array([np.random.random() for _ in range(0, N_time)])
    Q = np.array([np.random.random() for _ in range(0, N_time)])

    # Hop if and only if random number in P is greater than Boltzmann factor
    H1 = (P < pBoltz)

    # Hop anti-clockwise if random number in Q is less than 0.5, otherwise clockwise
    H = ['False'] * N_time
    for t in range(0, N_time):
        H[t] = H1[t] - 2.0 * ((Q[t] <= 0.5) and (H1[t] == 1))

    # Add together elements in vector
    theta = deltaTheta * np.cumsum(H)

    # Convert to Cartesian coordinates
    Rx = np.zeros((N_atoms, N_time))
    Ry = np.zeros((N_atoms, N_time))
    for w in range(0, N_atoms):
        Rx[w, :], Ry[w, :] = util.pol2cart(theta + (float(w) * deltaThetaAtom), radius)

    return Rx, Ry

def scattering_calculation(N_momentum, phi, Kxrange, Kyrange, N_atoms, Rx, Ry, coherent, N_time, dt, nfit):
    '''
    function scattering_calculation
    -------------------------------
    evaluates the intermediate scattering functions, fitting them to exponentials and evaluating the exponential decay constant for given momentum transfer
    :param N_momentum: number of momentum states (Kx, Ky)
    :param phi: scan angle
    :param Kxrange: range of momentum x-components
    :param Kyrange: range of momentum y-components
    :param N_atoms: number of atoms hopping
    :param N_time: number of hops in random walk
    :param dt: size of time step
    :param n_fit: number of time steps included in fit
    :param Rx: x-coordinates for each atom in each time step
    :param Ry: y-coordinates for each atom in each time step
    :param coherent: True if coherent scattering, False if incoherent scattering
    :return:
    '''

    #enumerate time steps
    time_seconds = int(N_time * dt)
    tscale = np.array(range(0, time_seconds))               # Timescale vector
    tscale = tscale / float(N_time)
    deltaT = tscale[1] - tscale[0]
    tscalefit = tscale[0:nfit]

    #initialise scattering function
    if N_atoms < 2:
        Rx = Rx[0]
        Ry = Ry[0]


    #convert scan angle to radians
    phi = phi * math.pi / 180.0

    #do momentum scan
    decay = np.zeros(N_momentum)  #to hold exponential decay constants
    momentum_transfer = np.zeros(N_momentum)  #to hold momentum transfer values
    EISF = np.zeros(N_momentum)   #to hold EISF
    QEA = np.zeros(N_momentum)    #to hold QEA
    for iK in range(0, N_momentum):
        K_x = (Kxrange[0] + (Kxrange[1] - Kxrange[0]) * float(iK - 1)/float(N_momentum - 1)) * math.cos(phi)
        K_y = (Kyrange[0] + (Kyrange[1] - Kyrange[0]) * float(iK - 1)/float(N_momentum - 1)) * math.sin(phi)

        # Calculate the time-dependent scattering function A
        A = np.zeros((N_atoms, N_time), dtype=np.complex)
        if N_atoms > 1:
            for v in range(0, N_atoms):
                for t in range(0, N_time):
                    A[v, t] = math.cos((K_x * Rx[t]) + (K_y * Ry[t])) + 1j * math.sin((K_x * Rx[t]) + (K_y * Ry[t]))
        else:
            A = A[0]
            for t in range(0, N_time):
                A[t] = math.cos((K_x * Rx[t]) + (K_y * Ry[t])) + 1j * math.sin((K_x * Rx[t]) + (K_y * Ry[t]))

        #for coherent scattering, sum scattering function over atoms
        if coherent and N_atoms > 1:
            A = sum(A, axis=0)  #sum over all atoms


        # calculate intermediate scattering function I
        FTA = np.zeros((A.shape))
        IFTA = np.zeros((A.shape))
        I = np.zeros(N_time)
        if coherent or N_atoms < 2:
            FTA = np.fft.fft(A)
            FTA_sq = np.multiply(FTA, np.conj(FTA))
            I1 = np.fft.ifft(FTA_sq) / float(N_time)
            I = np.real(I1) / max(np.real(I1))
        else:
            for l in range(0, N_atoms):
                FTA[l,:] = np.fft.fft(A[l,:])
                FTA_sq[l,:] = np.multiply(FTA[l,:], np.conj(FTA[l,:]))
                IFTA[l,:] = np.fft.ifft(FTA_sq[l,:])
            I1 = sum(IFTA) / float(N_time)
            I = np.real(I1) / max(np.real(I1))

        # Fit exponential to ISF
        Ifit = I[0:nfit]
        param_initial = [1.0, -2000]
        best_params, covar = spopt.curve_fit(util.exp_func, tscalefit, Ifit, p0=param_initial)
        a = best_params[0]
        b = best_params[1]

        decay[iK] = -b               # decay constant of intermediate scattering function
        momentum_transfer[iK] = math.sqrt(K_x * K_x + K_y * K_y)  # momentum transfer
        EISF[iK] = 1.0 - a           # offset is the elastic incoherent structure factor
        QEA[iK] = a                  # extinction efficiency factor


    return decay, momentum_transfer, EISF, QEA

def discrete_Markov_scattering(N_time=150000, delta_t=0.1, N_pots=6, phi=30, pBoltz=0.02, radius=2.0, N_atoms=1, N_momentum=100, nfit=100, coherent=False):
    '''
    function: discrete_Markov_scattering
    ------------------------------------
    simulates discrete rotational hopping of molecule on surface

    :param: N_time: number of time steps
    :param: dt: time step (picoseconds)
    :param: N_pots: lattice symmetry
    :param: pBoltz: hopping (Boltzmann) probability
    :param: radius: rotation radius
    :param: N_M: number of atoms in molecule
    :param: N_momentum: number of sampling points along each axis in momentum space
    :param: nfit: number of time steps used in exponential fit of intermediate scattering functions
    :param: coherent: coherent (True) or incoherent (False) scattering
    :return: success (True) or failure (False)
    '''

    #check symmetry
    if N_pots not in [1, 2, 3, 4, 6]:
        print('algorithm currently only implemented for hexagonal, triagonal and square lattice surfaces; slice = ' + str(slice) + ' conforms to none of these symmetries')

    #initialise momentum ranges
    Kxrange = np.array([0.11, 5])        # Range of K_x
    Kyrange = np.array([0.11, 5])        # Range of K_y

    #generate trajectory (rotational random walk)
    Rx, Ry = discrete_trajectory(N_time, N_atoms, pBoltz, N_pots, radius)

    #calculation of intermediate scattering function for each time step
    #if slice == 3 or slice == 6, we'd expect phi = 0 or phi = 30 degrees
    #if slice == 2 or slice == 4, we'd expect phi = 0 or phi = 45 degrees
    decay, momentum, EISF, QEA, = scattering_calculation(N_momentum, phi, Kxrange, Kyrange, N_atoms, Rx, Ry, coherent, N_time, delta_t, nfit)

    #normalise decay factor
    decay = decay / max(decay)

    #plot results for direction B
    plot_results(momentum, decay, EISF, QEA, N_pots, phi)

    return True

def continuous_Langevin_scattering(mass=2.0, radius=3.0, N_atoms=1, delta_t=0.1, N_time=500000, N_momentum=100, nfit=100, T=800, eta=3.0, V0=5.0, N_pots=6, phi=30.0, coherent=False):
    '''
    function: continuous_Langevin_scattering
    ----------------------------------------
    simulates continuous rotational motion of molecule on surface

    :param mass: mass of particle
    :param radius: radius of rotation
    :param N_atoms: number of atoms in molecule
    :param delta_t: size of time step
    :param N_time: number of time steps
    :param N_momentum: number of steps in momentum space
    :param nfit: number of time steps for which to fit exponential to intermediate scattering function
    :param T: temperature
    :param eta: friction coefficient
    :param V0: size of voltage peaks
    :param N_pots: number of voltage peaks around circle
    :param coherent: coherent (True) or incoherent (False) scattering

    :return: success (True) or failure (False)
    '''

    # Langevin rotational simulation
    # 30th June 2008

    ## Simulation Parameters
    Kxrange = [0.11, 5]   # Range of K_x
    Kyrange = [0.11, 5]   # Range of K_y

    '''
    TODO: scaling
    # Calculate properties in standard units
    #k = 1.3806503E-23     #Boltzmann constant in m^2 kg s^-2 K^-1 (SI units)
    #amu = 1.66e-27      # in kg (SI units)
    #dt_SI = dt/1e-12               # Time from ps to s
    #m_SI = mass * amu                   # Mass to kg
    #r_SI = radius * 1e-10                 # Radius to m
    ####I_SI = m_SI*r_SI^2            # Moment of inertia
    #I_SI = m_SI                    # For now as I_SI does not work
    #V0_SI = V0 * 1.602e-19 / 1000      # Potential to J
    '''

    ## Scattering Calculation for A-Direction
    Rx, Ry = continuous_trajectory(N_time, delta_t, N_atoms, N_pots, V0, eta, mass, radius, T)

    #calculation of intermediate scattering function for each time step
    #if N_pots == 3 or N_pots == 6, we'd expect phi = 0 or phi = 30 degrees
    #if N_pots == 2 or N_pots == 4, we'd expect phi = 0 or phi = 45 degrees
    decay, momentum, EISF, QEA, = scattering_calculation(N_momentum, phi, Kxrange, Kyrange, N_atoms, Rx, Ry, coherent, N_time, delta_t, nfit)

    #normalise decay factor
    decay = decay / max(decay)

    #plot results
    plot_results(momentum, decay, EISF, QEA, N_pots, phi)

    return True


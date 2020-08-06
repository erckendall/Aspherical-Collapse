import time
import sys
import numpy as np
import numexpr as ne
import numba
import pyfftw
import h5py
import os
from IPython.core.display import clear_output
import inspect

hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
light_year = 9.4607e15  # m
solar_mass = 1.989e30  # kg
axion_mass = 1e-23 * 1.783e-36  # kg  Note: Smaller assumption for axion mass
G = 6.67e-11  # N m^2 kg^-2
omega_m0 = 0.3111
omega_lambda = 0.6889
H_0 = 67.66 * 1e3 / (parsec * 1e6)  # s^-1


length_unit = (8 * np.pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25
time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** -0.5
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G)

####################### FUNCTION TO GENERATE PROGRESS BAR

def display_time(seconds, granularity=2):
    result = []
    intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append('{} {}'.format(value, name))
    return ', '.join(result[:granularity])

def prog_bar(iteration_number, progress, tinterval):
    size = 50
    status = ""
    relprogress = float(progress) / float(iteration_number)
    if relprogress >= 1.:
        relprogress, status = 1, "\r\n"
    block = int(round(size * relprogress))
    text = "\r[{}] {:.0f}% {}{}{}{}{}{}".format(
        "-" * block + " " * (size - block), round(relprogress * 100, 0),
        status, ' The previous step took ', tinterval, ' seconds.', ' Estimated time remaining is ', display_time( (iteration_number-progress)*tinterval, 5 ))
    sys.stdout.write(text)
    sys.stdout.flush()


####################### FUNCTION TO CONVERT TO DIMENSIONLESS UNITS

def convert(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value / length_unit
        elif (unit == 'km'):
            converted = value * 1e3 / length_unit
        elif (unit == 'pc'):
            converted = value * parsec / length_unit
        elif (unit == 'kpc'):
            converted = value * 1e3 * parsec / length_unit
        elif (unit == 'Mpc'):
            converted = value * 1e6 * parsec / length_unit
        elif (unit == 'ly'):
            converted = value * light_year / length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted = value / mass_unit
        elif (unit == 'solar_masses'):
            converted = value * solar_mass / mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value * solar_mass * 1e6 / mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted = value / time_unit
        elif (unit == 'yr'):
            converted = value * 60 * 60 * 24 * 365 / time_unit
        elif (unit == 'kyr'):
            converted = value * 60 * 60 * 24 * 365 * 1e3 / time_unit
        elif (unit == 'Myr'):
            converted = value * 60 * 60 * 24 * 365 * 1e6 / time_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value * time_unit / length_unit
        elif (unit == 'km/s'):
            converted = value * 1e3 * time_unit / length_unit
        elif (unit == 'km/h'):
            converted = value * 1e3 / (60 * 60) * time_unit / length_unit
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted

####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS

def convert_back(value, unit, type):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value * length_unit
        elif (unit == 'km'):
            converted = value / 1e3 * length_unit
        elif (unit == 'pc'):
            converted = value / parsec * length_unit
        elif (unit == 'kpc'):
            converted = value / (1e3 * parsec) * length_unit
        elif (unit == 'Mpc'):
            converted = value / (1e6 * parsec) * length_unit
        elif (unit == 'ly'):
            converted = value / light_year * length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted = value * mass_unit
        elif (unit == 'solar_masses'):
            converted = value / solar_mass * mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value / (solar_mass * 1e6) * mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted = value * time_unit
        elif (unit == 'yr'):
            converted = value / (60 * 60 * 24 * 365) * time_unit
        elif (unit == 'kyr'):
            converted = value / (60 * 60 * 24 * 365 * 1e3) * time_unit
        elif (unit == 'Myr'):
            converted = value / (60 * 60 * 24 * 365 * 1e6) * time_unit
        else:
            raise NameError('Unsupported time unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'm/s'):
            converted = value / time_unit * length_unit
        elif (unit == 'km/s'):
            converted = value / (1e3) / time_unit * length_unit
        elif (unit == 'km/h'):
            converted = value / (1e3) * (60 * 60) / time_unit * length_unit
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted


####################### FUNCTION TO SAVE OUTPUTS (COMOVING COORDINATES)

# 0: rho,  1: psi,  2: z-plane  4: yz-line,  5: x-plane,  6: y-plane,  7: xy-line,  8: xz-line,  9: psi-plane-x,  10: psi-plane-y,  11: psi-plane-z
# note: save_options[3] (energy) not in save_grid - calculated separately


def save_grid(rho, rfft_rho, psi, resol, save_options, npy, npz, hdf5, loc, ix, skip_saves, scalefactor):

        save_num = int((ix + 1) / (skip_saves+1))

        if (save_options[0]):
            if (npy):
                file_name = "rho_#{0}.npy".format(save_num)
                np.save(os.path.join(os.path.expanduser(loc), file_name),rho)
            if (npz):
                file_name = "rho_#{0}.npz".format(save_num)
                np.savez(os.path.join(os.path.expanduser(loc), file_name),rho)
            if (hdf5):
                file_name = "rho_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=rho)
                f.close()
        if (save_options[2]):
            planez = rho[:, :, int(resol / 2)-1]
            if (npy):
                file_name = "planez_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planez
                )
            if (npz):
                file_name = "planez_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planez
                )
            if (hdf5):
                file_name = "planez_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=planez)
                f.close()
        if (save_options[5]):
            planex = rho[int(resol / 2)-1, :, :]
            if (npy):
                file_name = "planex_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planex
                )
            if (npz):
                file_name = "planex_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planex
                )
            if (hdf5):
                file_name = "planex_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=planex)
                f.close()
        if (save_options[6]):
            planey = rho[:, int(resol / 2)-1, :]
            if (npy):
                file_name = "planey_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planey
                )
            if (npz):
                file_name = "planey_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    planey
                )
            if (hdf5):
                file_name = "planey_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=planey)
                f.close()
        if (save_options[1]):
            if (npy):
                file_name = "psi_#{0}.npy".format(save_num)
                np.save(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (npz):
                file_name = "psi_#{0}.npz".format(save_num)
                np.savez(
                    os.path.join(os.path.expanduser(loc), file_name),
                    psi
                )
            if (hdf5):
                file_name = "psi_#{0}.hdf5".format(save_num)
                file_name = os.path.join(os.path.expanduser(loc), file_name)
                f = h5py.File(file_name, 'w')
                dset = f.create_dataset("init", data=psi)
                f.close()
        if (save_options[4]):
            lineyz = rho[:, int(resol / 2)-1, int(resol / 2)-1]
            file_name = "lineyz_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                lineyz
            )
        if (save_options[7]):
            linexy = rho[int(resol / 2)-1, int(resol / 2)-1, :]
            file_name = "linexy_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                linexy
            )
        if (save_options[8]):
            linexz = rho[int(resol / 2)-1, :, int(resol / 2)-1]
            file_name = "linexz_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                linexz
            )
        if (save_options[9]):
            psiplanex = psi[int(resol / 2)-1, :, :]
            file_name = "psiplanex_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                psiplanex
            )
        if (save_options[10]):
            psiplaney = psi[:, int(resol / 2)-1, :]
            file_name = "psiplaney_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                psiplaney
            )
        if (save_options[11]):
            psiplanez = psi[:, :, int(resol / 2)-1]
            file_name = "psiplanez_#{0}.npy".format(save_num)
            np.save(
                os.path.join(os.path.expanduser(loc), file_name),
                psiplanez
            )


################################################EDGES FUNCTION

def edges_simp(psi, resol):

    for i in np.arange(resol):
        for j in np.arange(resol):
                psi[i,j,0] = 0
                psi[i, j, resol - 1] = 0
                psi[i,0,j] = 0
                psi[i,resol - 1,j] = 0
                psi[0, i, j] = 0
                psi[resol - 1, i, j] = 0

    return psi


####################### FUNCTION TO CALCULATE PHYSICAL ENERGIES (i.e. not comoving but still in code units unless conversion specified)

# Suspect that too many applications of the scalefactor - should only occur in distance measures i.e. start with comoving quantities then convert to physical
# Where are these 3 factors of the scale factor coming from in the wavefunction - all things are calculated in dimensionless code  units - no distance units come into the wavefunction

def calculate_energies(save_options, resol, psi, cmass, distarray, Vcell, phisp, karray2, funct, fft_psi, ifft_funct, egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, scalefactor, scalefactorlist,):

        if (save_options[3]):

            egyarr = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')


            # Gravitational potential energy density associated with the central potential
            egyarr = ne.evaluate('real((abs(psi))**2)') #compute physical density
            egyarr = ne.evaluate('real(-cmass/(distarray*scalefactor)*egyarr)') #compute physical energy density
            egpcmlist.append(Vcell * np.sum(egyarr)) #add total gravitational energy to the list
            tot = Vcell * np.sum(egyarr) #add total gravitational energy to the total energy


            # Gravitational potential energy density of self-interaction of the condensate
            egyarr = ne.evaluate('real(0.5*(phisp)*real((abs(psi))**2))')
            egpsilist.append(Vcell * np.sum(egyarr))
            tot = tot + Vcell * np.sum(egyarr)


            funct = fft_psi(psi)
            funct = ne.evaluate('-karray2*scalefactor**(-2.)*funct')
            funct = ifft_funct(funct)
            egyarr = ne.evaluate('real((-0.5*conj(psi)*funct))')
            ekandqlist.append(Vcell * np.sum(egyarr))
            tot = tot + Vcell * np.sum(egyarr)


            egylist.append(tot) #Physical total energy


            egyarr = ne.evaluate('real(((abs(psi))**2)/scalefactor**3)')
            mtotlist.append(Vcell* scalefactor**3 * np.sum(egyarr))
            
            scalefactorlist.append(scalefactor)


####################### FUNCTIONS TO CONVERT BETWEEN SCALE FACTOR / REDSHIFT / PROPER TIME (CODE UNITS)

## Checked - conversion to code units seems  fine
#
#
#
#
#def t_to_z(t):#t in code units
#    z = (np.sqrt(omega_m0/omega_lambda)*np.sinh( t*np.sqrt(omega_lambda/omega_m0*6*np.pi) ) )**(-2/3)-1
#    return z
#
#def z_to_t(z):#t in code units
#    t = np.sqrt(omega_m0/(omega_lambda*6*np.pi))*np.arcsinh( np.sqrt(omega_lambda/omega_m0)*(1+z)**(-3/2) )
#    return t
#
#def t_to_a(t):#t in code units
#    a = ( np.sqrt(omega_m0/omega_lambda)*np.sinh( t*np.sqrt(omega_lambda/omega_m0*6*np.pi) ) )**(2/3)
#    return a
#
#def a_to_t(a):#t in code units
#    t = np.sqrt(omega_m0/omega_lambda/6/np.pi)*np.arcsinh( np.sqrt(omega_lambda/omega_m0)*a**(3/2) )
#    return t


###################### CHOOSING THE TIMESTEP - I THINK THIS SHOULD BE IN DIMENSIONLESS COMOVING COORDINATES, SO SCALE FACTOR SHOULD NOT ENTER HERE


def choose_timestep(scalefactor, gridlength, resol, step_factor,):

    h = step_factor * scalefactor * (gridlength/float(resol))**2/np.pi
    return h
    
    
######################### FUNCTION TO INITIALIZE SOLITONS AND EVOLVE

def evolve(central_mass, num_threads, length, length_units, resol, initial_z, final_z, step_factor, skip_saves, save_options,
           save_path, npz, npy, hdf5, initfunct, c_mass_unit):
    print ('Initialising...')



    ##########################################################################################
    #SET INITIAL CONDITIONS
    
    t_0 = .001

    if (length_units == ''):
        gridlength = length #comoving grid length
    else:
        gridlength = convert(length, length_units, 'l')
        
    t = t_0*((1+initial_z)/(1+final_z))**(3/2) - t_0 #time duration  # Not a good way to define time if you want to run with no expansion - in that case redshift doesn't change
    timeparameter = t_0 #initial time in code units
    scalefactor = 1/(1+initial_z)
    redshift = initial_z
    
    if (c_mass_unit == ''):
        cmass = central_mass
    else:
        cmass = convert(central_mass, c_mass_unit, 'm')

    Vcell = (gridlength / float(resol)) ** 3 #comoving grid volume

    ne.set_num_threads(num_threads)

    ##########################################################################################
    # CREATE THE TIMESTAMPED SAVE DIRECTORY AND CONFIG.TXT FILE

    save_path = os.path.expanduser(save_path)
    tm = time.localtime()

    talt = ['0', '0', '0']
    for i in range(3, 6):
        if tm[i] in range(0, 10):
            talt[i - 3] = '{}{}'.format('0', tm[i])
        else:
            talt[i - 3] = tm[i]
    timestamp = '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(tm[0], '.', tm[1], '.', tm[2], '_', talt[0], ':', talt[1], ':', talt[2], '_', resol)
    file = open('{}{}{}'.format('./', save_path, '/timestamp.txt'), "w+")
    file.write(timestamp)
    os.makedirs('{}{}{}{}'.format('./', save_path, '/', timestamp))
    file = open('{}{}{}{}{}'.format('./', save_path, '/', timestamp, '/config.txt'), "w+")
    file.write(('{}{}'.format('resol = ', resol)))
    file.write('\n')
    file.write(('{}{}'.format('axion_mass (kg) = ', axion_mass)))
    file.write('\n')
    file.write(('{}{}'.format('length (code units) = ', gridlength)))
    file.write('\n')
    file.write(('{}{}'.format('duration (code units) = ', t)))
    file.write('\n')
    file.write(('{}{}'.format('step_factor  = ', step_factor)))
    file.write('\n')
    file.write(('{}{}'.format('initial redshift  = ', initial_z)))
    file.write('\n')
    file.write(('{}{}'.format('final redshift  = ', final_z)))
    file.write('\n')
    file.write(('{}{}'.format('central_mass (code units) = ', cmass)))
    file.write('\n\n')
    file.write(('{}{}'.format('function source code: \n', inspect.getsource(initfunct))))
    file.write(('{}{}'.format('\ns_mass_unit = ', c_mass_unit)))
    file.write('\n\nNote: If the above unit is blank, this means that the mass parameter was specified in code units')
    file.close()

    loc = save_path + '/' + timestamp



    ##########################################################################################
    # SET UP THE REAL SPACE COORDINATES OF THE GRID

    gridvec = np.linspace(-gridlength / 2.0 + gridlength / float(2 * resol), gridlength / 2.0 - gridlength / float(2 * resol), resol)
    xarray, yarray, zarray = np.meshgrid(
        gridvec, gridvec, gridvec,
        sparse=True, indexing='ij',
    )
    distarray = ne.evaluate("(xarray**2+yarray**2+zarray**2)**0.5") # Radial coordinates



    ##########################################################################################
    # SET UP K-SPACE COORDINATES FOR COMPLEX DFT (NOT RHO DFT)

    kvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    kxarray, kyarray, kzarray = np.meshgrid(
        kvec, kvec, kvec,
        sparse=True, indexing='ij',
    )
    karray2 = ne.evaluate("kxarray**2+kyarray**2+kzarray**2")



    ##########################################################################################
    # INITIALISE INITIAL DENSITY FIELD
    
    
    psi = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')
    funct = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')
    
    fft_psi = pyfftw.builders.fftn(psi, axes=(0, 1, 2), threads=num_threads)
    ifft_funct = pyfftw.builders.ifftn(funct, axes=(0, 1, 2), threads=num_threads)
    
    #custom density:
    field = pyfftw.zeros_aligned((resol, resol, resol), dtype='complex128')   
    field = initfunct(xarray, yarray, zarray)
    psi = ne.evaluate("(psi + field)")
    psi = edges_simp(psi, resol)

    rho = ne.evaluate("real(abs(psi)**2)")
    
    
    ##########################################################################################
    #COMPUTE NUMBER OF TIMESTEPS PERFORMED
    
    tdummy = t_0 
    num_timesteps = 0
    while(tdummy < t_0*((1+initial_z)/(1+final_z))**(3/2)):
        num_timesteps+=1
        tdummy += choose_timestep((1/(1+initial_z))*(tdummy/t_0)**(2/3), gridlength, resol, step_factor, )

    
    ##########################################################################################
    # SETUP K-SPACE FOR RHO (REAL)

    rkvec = 2 * np.pi * np.fft.fftfreq(resol, gridlength / float(resol))
    krealvec = 2 * np.pi * np.fft.rfftfreq(resol, gridlength / float(resol))
    rkxarray, rkyarray, rkzarray = np.meshgrid(
        rkvec, rkvec, krealvec,
        sparse=True, indexing='ij'
    )

    rkarray2 = ne.evaluate("rkxarray**2+rkyarray**2+rkzarray**2")

    rfft_rho = pyfftw.builders.rfftn(rho, axes=(0, 1, 2), threads=num_threads)
    phik = rfft_rho(rho)  # not actually phik but phik is defined in next line
    phik = ne.evaluate("-4*3.141593*phik/(rkarray2*scalefactor)")
    phik[0, 0, 0] = 0
    irfft_phi = pyfftw.builders.irfftn(phik, axes=(0, 1, 2), threads=num_threads)



    ##########################################################################################
    # COMPUTE INTIAL VALUE OF POTENTIAL

    phisp = pyfftw.zeros_aligned((resol, resol, resol), dtype='float64')
    phisp = irfft_phi(phik)
    phisp = ne.evaluate("phisp-(cmass)/(distarray)") # Again, should be working in dimensionless covariant code units, then converting to physicals at the end


    ##########################################################################################
    # PRE-LOOP ENERGY CALCULATION

    if (save_options[3]):
        egylist = []
        egpcmlist = []
        egpsilist = []
        ekandqlist = []
        mtotlist = []
        scalefactorlist = []


        calculate_energies(
            save_options, resol,
            psi, cmass, distarray, Vcell, phisp, karray2, funct,
            fft_psi, ifft_funct,
            egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, scalefactor, scalefactorlist,
        )


    ##########################################################################################
    # PRE-LOOP SAVE I.E. INITIAL CONFIG
    save_grid(
            rho, rfft_rho, psi, resol,
            save_options,
            npy, npz, hdf5,
            loc, -1, 0, scalefactor,
    )  ## ix = -1, and skip_saves = 0 therefore save_num = 0


    ##########################################################################################
    # LOOP NOW BEGINS



    # Note: Has lost initial and final half-step - therefore not technically 2nd order convergent
#Needs to all be full steps if tilmestep changing
    clear_output()
    print("The total number of steps is %.0f" % num_timesteps)
    print('\n')
    tinit = time.time()
    save_counter = 0
    
    a_0 = 1/(1+initial_z)
    
    for ix in range(num_timesteps):

        h = choose_timestep(scalefactor, gridlength, resol, step_factor,)
        psi = ne.evaluate("exp(-0.5*1j*h*phisp)*psi")
        funct = fft_psi(psi)
        funct = ne.evaluate("exp(-1j*h*karray2*(scalefactor**(-2))*0.5)*funct")
        psi = ifft_funct(funct)
        rho = ne.evaluate("real(abs(psi)**2)")
        phik = rfft_rho(rho)  
        phik = ne.evaluate("-4*3.141593*(phik)/(rkarray2*scalefactor)")
        phik[0, 0, 0] = 0
        phisp = irfft_phi(phik)
        phisp = ne.evaluate("phisp-(cmass)/(distarray)") # And again
        psi = ne.evaluate("exp(-0.5*1j*h*phisp)*psi")
	psi = edges_simp(psi, resol)

        ##############################
        #UPDATE TIMEPARAMETER, SCALEFACTOR AND REDSHIFT
            #The evolved quantities get saved at the new scalefactor

        timeparameter+=h
	# print timeparameter, a_0 * (timeparameter/float(t_0))**(2/float(3))
        scalefactor = a_0 * (timeparameter/float(t_0))**(2/float(3))
        redshift = 1/scalefactor - 1
        
        #Next if statement ensures that an extra half step is performed at each save point
        if (save_counter==skip_saves):

            #Next block calculates the energies at each save, not at each timestep.
            calculate_energies(
                save_options, resol,
                psi, cmass, distarray, Vcell, phisp, karray2, funct,
                fft_psi, ifft_funct,
                egpcmlist, egpsilist, ekandqlist, egylist, mtotlist, scalefactor, scalefactorlist, )
            #Next block saves the grid at each save, not at each timestep.
            save_grid(
                    rho, rfft_rho, psi, resol,
                    save_options,
                    npy, npz, hdf5,
                    loc, ix, skip_saves, scalefactor,
            )
            save_counter=0
        else:
            save_counter+=1

        
        ################################################################################
        # UPDATE INFORMATION FOR PROGRESS BAR

        tint = time.time() - tinit
        tinit = time.time()
        prog_bar(num_timesteps, ix + 1, tint)



    ################################################################################
    # LOOP ENDS

    clear_output()
    print ('\n')
    print("Complete.")

    if (save_options[3]):
        file_name = "egylist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egylist)
        file_name = "egpcmlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpcmlist)
        file_name = "egpsilist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), egpsilist)
        file_name = "ekandqlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), ekandqlist)
        file_name = "masslist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), mtotlist)
        file_name = "scalefactorlist.npy"
        np.save(os.path.join(os.path.expanduser(loc), file_name), scalefactorlist)

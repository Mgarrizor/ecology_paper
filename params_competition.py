import numpy as np
import argparse


parser = argparse.ArgumentParser()

# ======== No default value ==============================
parser.add_argument("-i",  "--indx_decay",                      type = int,   help="Index of the decay rate vector for phage #2 ; from 0 - 19 (strong --> weak)")

# ======== With default value ============================
parser.add_argument("-M",  "--hgrid",           default = 1000,    type = int,   help="Number of horizontal grid points ; Must be an even number")
parser.add_argument("-L",  "--sig_length",      default = 1,       type = float, help="Signal length")
parser.add_argument("-np", "--n_processes",     default = 1,       type = int,   help="Number of processes for MPI")
parser.add_argument("-it", "--int_time",        default = 350,     type = float, help="Integration time, in years")
parser.add_argument("-f",  "--freq",            default = 1e3,     type = int,   help="Frequency of output in number of time steps")
parser.add_argument("-n",  "--name",            default = "test",  type = str,   help="Name of parent folder")
parser.add_argument("-t",  "--sim_flag",        default = 1,       type = int,   help="Write down population density trajectories: 0 = False, 1 = True")



args = parser.parse_args()

M         = args.hgrid
N         = 2
i         = args.indx_decay
processes = args.n_processes		# If > 1 then launch parallel simulations, each of which takes a different (j0,p) pair
sim_flag  = bool(args.sim_flag)

#=================================================================#

integration_time = args.int_time*365       # In days
#================#

freq      = args.freq          # Writting frequency in number of time steps
ϵ         = False              # Activate/Deactivate (True/False) writting threshold flag
threshold = int(2e5)           # Threshold flag -- Number of final steps where we start saving the trajectories
#================#
signal        = True	                # If True correct decay rate in the upper layer to fulfill the signal length
signal_length = args.sig_length         # Length in terms of the fraction of the total system
# Initial condition flags 

random_loguni    = False    # Loguniform distributed initial densities
random_uni       = True     # Uniformly distributed initial densities

heat_map    = True
high_res    = False
single_test = False
#=================================================================#

name = args.name     # Name of the file under which simulation results will be stored.

#======== These flags should NOT be touched =================================#

#===Population dynamics System parameters ***********************************

K_0 = 1e12                    # Carrying capacity, 1e6 /mL from --> [1]

g   = 0.5/(3600*24)           # Growth rate, 2 fissions per day from --> [2]
							  #-----------------------------------------------------------------------------------
δ_s = 0.005/3600              # Decay rate surface      from [3] (see References at the end of this document)
							  #-----------------------------------------------------------------------------------							   	
δ_a = 0.01/60                 # Decay rate atmosphere   for bacteria from --> [4]
							  #                         for phages   from --> [5]
							  #-----------------------------------------------------------------------------------
β   = 100                     # Burst size from --> [3]
η   = 100.*1e-15/3600         # Adsorption rate 12 1e-9 mL/h from --> [3]


# Competition experiment ==============================

δ_up  = η*(β-1)*K_0
δ_dummy   = np.logspace(-5,0,20)*δ_up

δ_s   = np.logspace(1,3,10)*δ_dummy[0]
δ_vec = [δ_dummy[0],δ_s[i]]

#*******************************************************************************************************#
#=============== Space parameters ======================================================================#

periodic_bc = False

Dx = 0.188*1e-12      # (m^2 / s)   # Rod-shaped bacterium's (E.Coli)
Dy = 0.154*1e-12      # (m^2 / s)   # transversal and longitudinal diffusion coeff.
Db = np.mean([Dx,Dy])               # from --> [4]

Dp = 2.76*1e-12 # m^2 / s   # Phage diffusion coefficient, from -->  [5]

Db, Dp = 0, 0               # By default

dt  = 50     # (s)          # Time step
v_0 = 1      # (m/s)        # Advection velocity
dx  = 1*dt   # (m)          # Grid size (Courant number = 1 imposed)
#N   = 2                     # ygrid
#M   = 500                   # xgrid

K   = np.ones(M)*K_0 		  # Set to be constant in space
tau = dt

#*******************************************************************************************************#
#=============== Signal length controlled by the upper layer's decay rate ==============================#
if signal:                                                
	factor        = 1/(M*signal_length*δ_a)*np.log(dx**3*K_0)/dt       # Correction factor to δ_a to accomplish the signal_length value
	δ_a           = factor*δ_a                                         # Apply correction factor

#*******************************************************************************************************#
#========================= Configurations ==============================================================#
#
# 1) Heat-map simulations 
if heat_map:

	processes = 100                     # Good number for cellX server is 125
	#integration_time = 400*365         # 400 years should sufficient to reach the steady-state
	flux = np.logspace(-2,-1,10)/tau    # 
	prob = np.logspace(-4,-3,10)     

	random_uni = True              
	#freq       = int(2e2)           
	ϵ          = False              
	threshold  = int(2e6) 			   # If only save the last (2 million / freq) time steps --> enough to get good statistics
  
#=======================================================#
#
# 2) High resolution simulations 

if high_res:

	processes = 10
	integration_time = 200*365
	flux = np.logspace(-5,0,10)/tau
	prob = [3e-5]

	random_uni = True              
	freq       = int(4e3)           # Save every 1e3 steps
	ϵ          = False              # Deactivated "threshold" flag
	threshold  = int(2e6)           # This flag is not used

#=======================================================#
#
# 3) Code test

if single_test:


	f = np.logspace(-5,0,20)/tau
	flux = [f[-4]]
	prob = [1e-4]

	random_uni = True              
#	freq       = int(1e3)           # Save every 1e3 steps
	ϵ          = False              # Deactivated "threshold" flag
	threshold  = int(2e6)           # This flag is not used

#*******************************************************************************************************#
# ======== References ==================================================================================#
#****
# [1]
#****
# [2] 
#****
# [3] De Paepe M, Taddei F. Viruses’ Life History: Towards a Mechanistic Basis of a Trade-Off between Survival and Reproduction among Phages. PLoS Biol 4(7): e193. https://doi.org/10.1371/journal.pbio.0040193 (2006).
#****
# [4] Kinetics and mechanism analysis on self-decay of airborne bacteria: biological and physical decay under different temperature (2022)
#****
# [5] Primary biological aerosol particles in the atmosphere: A review" (2012) 
#  
#      Read from “Even for the most stable viruses, k is typically of the order of 0.01 min−1” Section 2.4 Viruses.
#****
# [6] Tavaddod, S., Charsooghi, M. A., Abdi, F., Khalesifard, H. R. & Golestanian, R. Probing passive diffusion of flagellated and deflagellated Escherichia coli. The Eur. Phys.J. E 34, 16, (2011).
#****
# [7] R. Moldovan, E. Chapman-McQuiston, X.L. Wu, On Kinetics of Phage Adsorption, Biophysical Journal, Volume 93, Issue 1, 2007, Pages 303-315, ISSN 0006-3495, https://doi.org/10.1529/biophysj.106.102962.
#****
# [8] 






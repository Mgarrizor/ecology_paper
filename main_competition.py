import numpy as np
import time
import numba
import os, sys
#====== Import modules ===========#
import params_competition as par
import tstepping_competition as st
import ios
#=================================#
from scipy.stats import loguniform

from multiprocessing.sharedctypes import Value
from multiprocessing import Pool, Lock

from numba import prange
from tqdm.auto import trange
from numba.typed import List

from datetime import date

#=================================#
r = np.random
#=================================#
δ = List()

δ_sur = par.δ_vec
δ.append(δ_sur[0])
δ.append(δ_sur[1])
δ.append(par.δ_a)
#=================================#
def num_steps(Δt,tf):
    
    tf_seconds = tf*(24*60*60)
    
    return int(tf_seconds/Δt)
    
#=============================#
# These are the default options that will be used when writting the .txt files
np.set_printoptions(threshold=np.inf, formatter={'float': '{: 0.2e}'.format})
#=====================#
Nsteps = num_steps(par.dt,tf = par.integration_time)
#=====================#
ϵ = par.ϵ
if (ϵ == True):
    threshold = par.threshold  # Writting threshold
else:
    threshold = Nsteps
#=================================#
#=============================#
def integrate(zz, v_0 = par.v_0, η = par.η , g = par.g, K = par.K, δ = δ, β = par.β, Δt=par.dt, Δx = par.dx, N=par.N, M = par.M, steps= Nsteps, f = par.freq, periodic_bc = par.periodic_bc):
    
    seconds_0 = time.time() 
    
    p = zz[0]
    j = zz[1]
    
    path = os.getcwd()

    f = f   # Writes every "f" steps
    v = v_0          # Drift

    IΔx     = 1/Δx
    cut_off = IΔx**3 # Extinction cut-off
    
    # Lax-Wendroff Method - Linear Advection ===========#
    
    c = v*Δt/Δx # Courant Number
 
    if c > 1:
        
        print("Courant number too big!")
        return

    bm1 = 0.5*c*(1+c)     # Lax-Wendroff coefficients
    b0  = 1-c**2
    b1  = -0.5*c*(1-c)
    b = np.array([bm1,b0,b1])

    # Initialising the system ==========================#
    #---------------------------------#
    name = par.name
    root = ios.make_dir(path,name,j,p,δ)

    random_uni    = par.random_uni
    random_loguni = par.random_loguni

    # If atm = False:

    B_fixed_1 = δ[0]/(η*(β-1))               # Well-mixed steady-state
    P_fixed_1 = (1-B_fixed_1/par.K_0)*g/η

    B_fixed_2 = δ[1]/(η*(β-1))               # Well-mixed steady-state
    P_fixed_2 = (1-B_fixed_2/par.K_0)*g/η

    print(par.args)
    print(f"P*1 = {P_fixed_1:.1e} ; P*2 = {P_fixed_2:.1e}")
    print(f"δ_1 = {δ[0]:.1e} ; δ_2 = {δ[1]:.1e}  ; δ_a = {δ[2]:.1e} ")

    P1_0 = np.zeros([N,M])
    P2_0 = np.zeros([N,M])
    B_0  = np.zeros([N,M])

    # Initial density profiles ===================================#

    if random_loguni:  # Random population densities uniformly distributed on a log10 scale


        B_0[0][::2]  = loguniform.rvs(cut_off,B_fixed_1, size = int(M/2))   # M must be an even number
        B_0[0][::-2] = loguniform.rvs(cut_off,B_fixed_2, size = int(M/2))

        P1_0[0][::2]  = loguniform.rvs(cut_off,P_fixed_1, size = int(M/2))   
        P2_0[0][::-2] = loguniform.rvs(cut_off,P_fixed_2, size = int(M/2))

    elif random_uni:  # Random population densities uniformly distributed


        B_0[0][::2]  = r.uniform(cut_off,B_fixed_1,int(M/2))   # M must be an even number
        B_0[0][::-2] = r.uniform(cut_off,B_fixed_2,int(M/2))

        P1_0[0][::2]  = r.uniform(cut_off,P_fixed_1,int(M/2))   
        P2_0[0][::-2] = r.uniform(cut_off,P_fixed_2,int(M/2))

    #====================================================================#
    #====================================================================#

    B_t  = [B_0.copy()]
    P1_t = [P1_0.copy()]
    P2_t = [P2_0.copy()]
    t    = [0]

    counter = 0
    
    ios.ios_init(root,Δx,M,N,Δt,v,β,δ,η,g,j,p,steps,B_0,P1_0,P2_0) # Create output files and simulation parameters file
#===========================================================#   
#===========================================================#

    for _ in trange(steps):

        u1 = r.uniform(0,1,M)
        u2 = r.uniform(0,1,M)
        u3 = r.uniform(0,1,M)
        

        B_0, P1_0, P2_0 = st.update_LW_RK4(B_0,P1_0,P2_0,Δt,IΔx,N,η,g,K,δ,β,j,v,u1,u2,u3,p,b,c,periodic_bc)

        B_0[B_0<cut_off]   = 0
        P1_0[P1_0<cut_off] = 0
        P2_0[P2_0<cut_off] = 0

        counter += 1

        
        ios.ios_save(counter,f,steps,threshold,Δt,root,B_0,P1_0,P2_0,M)
        #******************************************************************************#
        if (np.sum(B_0[0]) + np.sum(B_0[1])) <= cut_off:

            # Write the reason of break
            with open (os.path.join(root,f'progress'+".txt"), "a") as outfile:

                outfile.write(f'Extinction --> B_0[0] = {np.sum(B_0[0]):.1e} ; B_0[1] = {np.sum(B_0[1]):.1e} \n')
                outfile.write(f'Stopping simulation at (fraction of) Simulated time = {counter/steps:.5f} \n')
                outfile.close()

            break
        #******************************************************************************#

    seconds_1 = time.time()                   
    Time = (seconds_1-seconds_0)/60

    with open (os.path.join(root,f'progress'+".txt"), "a") as outfile:
                outfile.write(f'========== Simulation finished ============= \n')
                outfile.write(f'# Real time = {Time:.2f} minutes')
                outfile.close()
    
    return 1

#================================#
#===================   Main    =============#
npro  = par.processes
p     = par.prob
j     = par.flux
pp,jj = np.meshgrid(p,j)
p     = pp.flatten()
j     = jj.flatten()
zz    = np.stack((p,j),axis = 1)

ptotal    = np.size(pp)
pfinished = 0

if __name__ == '__main__':
    
    control_0 = time.time()
    
    path = os.getcwd()
    root = f"{path}/"+par.name
    
    try:   
        os.mkdir(root)
    except Exception:
        pass
    
    with open (os.path.join(root,f'global_progress'+".txt"), "w") as outfile:
                outfile.write(f'Progress bar {0:.3f} %')
                outfile.close()
    
    with Pool(npro) as pl:     # Open pool of npro processes to compute in parallel
        
        for result in pl.imap_unordered(integrate, zz):     # Pick up the results without waiting for the rest to finish (return 1)
        
            pfinished += result                                       # Add the result to the counter to have a global progress bar
            print(f'{pfinished/ptotal*100:.2f} %', flush=True)        # Show via stdout as well as in file
            
            with open (os.path.join(root,f'global_progress'+".txt"), "w") as outfile:
                outfile.write(f'Progress bar {pfinished/ptotal*100:.3f} % ; Elapsed time {(time.time()-control_0)/60:.2f} minutes \n')
                outfile.close()
    
    d = date.today()
    d2 = d.strftime("%B %d, %Y")
    
    with open (os.path.join(root,f'global_progress'+".txt"), "a") as outfile:
                outfile.write(f'Elapsed time {(time.time()-control_0)/60:.2f} minutes ; {(time.time()-control_0)/(60*60):.2f} hours ; {(time.time()-control_0)/(60*60*24):.2f} days\n')
                outfile.write(f'{d2}')
                outfile.close()
                
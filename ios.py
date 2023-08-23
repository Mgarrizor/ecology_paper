import numpy as np
import os, sys
import params_competition as par



def make_dir(path,name,j,p,δ):

    root = f"{path}/"+name
    try:   
        os.mkdir(root)
    except Exception:
        pass
    #---------------------------------#
    root = f"{path}/"+name+f"/{j:.1e}"
    try:   
        os.mkdir(root)
    except Exception:
        pass
    #---------------------------------#
    root = f"{path}/"+name+f"/{j:.1e}/{p:.1e}"
    try:
        os.mkdir(root)
    except Exception:
        pass
    #---------------------------------#
    root = f"{path}/"+name+f"/{j:.1e}/{p:.1e}"+f"/{δ[0]:.1e}"
    try:   
        os.mkdir(root)
    except Exception:
        pass

    #---------------------------------#
    root = f"{path}/"+name+f"/{j:.1e}/{p:.1e}"+f"/{δ[0]:.1e}/{δ[1]:.1e}"
    try:
        os.mkdir(root)
    except Exception:
        pass
#---------------------------------#
    return root

def ios_init(root,Δx,M,N,Δt,v,β,δ,η,g,j,p,steps,B_0,P1_0,P2_0):
#    Create output file ; Write the system parameters
    with open (os.path.join(root,f'parameters'+".txt"), "w") as outfile:
            outfile.write(f'========================================================================== \n')
            outfile.write(f'Physical parameters:  dx = {Δx:.0f} m ; xgrid = {M:.0f} ; ygrid = {N:.0f} \n')
            outfile.write(f'                      dt = {Δt:.0f} s ; v     = {v:.0f} m/s \n')
            outfile.write(f'Phage parameters: beta = {β:.0f}; delta_s = {δ[0]:.1e} s-1; delta_a = {δ[1]:.1e} s-1 ; Absorption rate eta = {η:.1e} \n')
            outfile.write(f'Bacterial parameters: g = {g:.1e}  \n')
            outfile.write(f'========================================================================== \n')
            outfile.write(f'Fluxes: jB = {j:.2e} g ; jP = {j:.2e} g ; p = {p:.2e}         \n')
            outfile.write(f'========================================================================== \n')
         #   outfile.write(f'Initial conditions:      \n')
         #   outfile.write(f'                    P[0][0](0) = {P_0[0][0]:.2e}       \n')
         #   outfile.write(f'                    B[0][0](0) = {B_0[0][0]:.2e}       \n')
         #   outfile.write(f'========================================================================== \n')
            outfile.write(f'                    freq = {par.freq:.0f}       \n')
            outfile.write(f'                    random_uni  = {par.random_uni} ; random_loguni = {par.random_loguni}                    \n')
            outfile.write(f'                    signal_corr = {par.signal} ; L = {par.signal_length}                    \n')
            outfile.write(f'Integration time = {par.integration_time:.1f} days                                      \n')
            outfile.write(f'========================================================================== \n')
            outfile.close()

    # Progress "bar"
    with open (os.path.join(root,f'progress'+".txt"), "w") as outfile:
            outfile.write(f'Progress bar ; Days to be simulated = {steps*Δt/(60*60*24)} \n')
            outfile.close()

    time_t = 0        
    sim_flag = par.sim_flag
    print(f"Write down trajectories = {sim_flag}")
    print(f"Integration time        = {par.integration_time/365:.1f} yrs")

    if sim_flag:
        
        with open (os.path.join(root,f'sim'+".txt"), "w") as outfile:
            
            outfile.write(f'Trajectories: \n')
            outfile.write(f't_i, B[0](t_i) \n')
            outfile.write(f't_i, P1[0](t_i) \n')
            outfile.write(f't_i, P2[0](t_i) \n')

            outfile.write(str(time_t)+"  ")
            outfile.write(str(B_0[0]).strip("[]").replace('\n','').strip(" "))
            outfile.write("\n")
            outfile.write(str(time_t)+"  ")
            outfile.write(str(P1_0[0]).strip("[]").replace('\n','').strip(" "))
            outfile.write("\n")
            outfile.write(str(time_t)+"  ")
            outfile.write(str(P2_0[0]).strip("[]").replace('\n','').strip(" "))
            outfile.write("\n")
            outfile.close()

    # Fraction of columns in C,F,E states 
    with open (os.path.join(root,f'frac'+".txt"), "a") as outfile:


                dummy_a = B_0[0]
                dummy_b = P1_0[0]
                dummy_c = P2_0[0]
                zero, coex1, coex2, ones = len(dummy_a[dummy_a == 0])/M,len(dummy_a[(dummy_a != 0) & (dummy_b != 0)])/M,len(dummy_a[(dummy_a != 0) & (dummy_c != 0)])/M, len(dummy_a[(dummy_a != 0) & (dummy_b == 0) & (dummy_c == 0)])/M

                outfile.write(f't, Zero, Coexistence, Ones \n')
                outfile.write(str(time_t)+"  ")
                outfile.write(str(zero)+"  ")
                outfile.write(str(coex1)+"  ")
                outfile.write(str(coex2)+"  ")
                outfile.write(str(ones)+"  ")
                outfile.write("\n")

                outfile.close()

                del dummy_a
                del dummy_b
                del dummy_c


def ios_save(counter,f,steps,threshold,Δt,root,B_0,P1_0,P2_0,M):

    #******************************************************************************#
        
        if (counter%f == 0) & (counter >= steps-threshold):
            time_t = counter*Δt
            sim_flag = par.sim_flag
            if sim_flag:
                with open (os.path.join(root,f'sim'+".txt"), "a") as outfile:
                   outfile.write(str(time_t)+"  ")
                   outfile.write(str(B_0[0]).strip("[]").replace('\n','').strip(" "))
                   outfile.write("\n")
                   outfile.write(str(time_t)+"  ")
                   outfile.write(str(P1_0[0]).strip("[]").replace('\n','').strip(" "))
                   outfile.write("\n")
                   outfile.write(str(time_t)+"  ")
                   outfile.write(str(P2_0[0]).strip("[]").replace('\n','').strip(" "))
                   outfile.write("\n")

                   outfile.close() 

            with open (os.path.join(root,f'frac'+".txt"), "a") as outfile:

                dummy_a = B_0[0]
                dummy_b = P1_0[0]
                dummy_c = P2_0[0]
                zero, coex1, coex2, ones = len(dummy_a[dummy_a == 0])/M,len(dummy_a[(dummy_a != 0) & (dummy_b != 0)])/M, len(dummy_a[(dummy_a != 0) & (dummy_c != 0)])/M, len(dummy_a[(dummy_a != 0) & (dummy_b == 0) & (dummy_c == 0)])/M

                outfile.write(str(time_t)+"  ")
                outfile.write(str(zero)+"  ")
                outfile.write(str(coex1)+"  ")
                outfile.write(str(coex2)+"  ")
                outfile.write(str(ones)+"  ")
                outfile.write("\n")

                outfile.close()

                del dummy_a
                del dummy_b
                del dummy_c

        if counter%(int(0.1*steps))== 0:

            with open (os.path.join(root,f'progress'+".txt"), "a") as outfile:
                outfile.write(f'(fraction of) Simulated time = {counter/steps:.2f} \n')
                outfile.close()
        #******************************************************************************#














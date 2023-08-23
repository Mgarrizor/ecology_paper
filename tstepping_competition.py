import numpy as np
import numba

@numba.njit
def van_Leer(N):
    
    f = N - np.roll(N,1)
    div = np.roll(N,-1) - N

    case_1 = (div==0) & (f != 0)
    case_2 = (div==0) & (f == 0)

    f[case_1] = 0
    div[case_1] = 1

    f[case_2]   = 0
    div[case_2] = 1
    
    θ = f/div
    ϕ = (θ + np.abs(θ))/(1 + np.abs(θ))
    
    ϕ[case_1] = 2
    ϕ[case_2] = 0

    return ϕ
#=======================#
@numba.njit
def update_LW_RK4(B,P1,P2,Δt,IΔx,N,η,g,K,δ,β,j,v,u1,u2,u3,p,b,c,periodic_bc):
    
   # Sources-Sinks step ===================================================#

    RK_step_Bs, RK_step_Ba, RK_step_Ps, RK_step_Pa, RK_step_P2s, RK_step_P2a = coeff(g,K,η,δ,β,B,P1,P2,Δt,j,u1,u2,u3,p,IΔx)
    
   # Surface *****************

    B[0]  = B[0]  + RK_step_Bs 
    P1[0] = P1[0] + RK_step_Ps
    P2[0] = P2[0] + RK_step_P2s
    
    # Atmosphere *************
    
    B[1]  = B[1]  + RK_step_Ba      
    P1[1] = P1[1] + RK_step_Pa
    P2[1] = P2[1] + RK_step_P2a

    # Advection step ======================================================#
    
    #ϕ_B = van_Leer(B[1])
    #ϕ_P = van_Leer(P[1])
    
    if periodic_bc:
        
        B[1]  = lax_wendroff_periodic(B[1],b,c)   
        P1[1] = lax_wendroff_periodic(P1[1],b,c)
        P2[1] = lax_wendroff_periodic(P2[1],b,c)
        
    else:
    
        B[1]  = lax_wendroff_open(B[1],v,c)   
        P1[1] = lax_wendroff_open(P1[1],v,c)
        P2[1] = lax_wendroff_open(P2[1],v,c)
        
    return B,P1,P2

@numba.njit
def FB(Bs,Ps,Ps2,Ba,g,K,η,j,boolean_B,IΔx):

   # return D_B*laplacian_open(Bs)*IΔx**2 - η*Bs*Ps + g*Bs*(1 - Bs/K) + fluxes(Bs,Ba,j,boolean_B)
    return  - η*Bs*(Ps+Ps2) + g*Bs*(1 - Bs/K) + fluxes(Bs,Ba,j,boolean_B)

@numba.njit
def FBA(Ba,Bs,δ,j,boolean_B):

    return -δ*Ba - fluxes(Bs,Ba,j,boolean_B)

@numba.njit
def FP(Bs,Ps,Pa,β,δ,η,j,boolean_P,IΔx):

   # return D_P*laplacian_open(Ps)*IΔx**2 + (β-1)*η*Bs*Ps - δ*Ps + fluxes(Ps,Pa,j,boolean_P)
    return  (β-1)*η*Bs*Ps - δ*Ps + fluxes(Ps,Pa,j,boolean_P)

@numba.njit
def FPA(Pa,Ps,δ,j,boolean_P):
 
    return -δ*Pa - fluxes(Ps,Pa,j,boolean_P)

@numba.njit
def coeff(g,K,η,δ,β,B,P,P2,Δt,j,u1,u2,u3,p,IΔx):

    Bs,Ba   = B[0],B[1]
    Ps,Pa   = P[0],P[1]
    P2s,P2a = P2[0],P2[1]

    δs,δs2,δa = δ[0],δ[1],δ[2]
    #=============================#
    # RK4 Surface and Atmosphere
    boolean_B  = u1<p
    boolean_P  = u2<p
    boolean_P2 = u3<p
    # Coefficients ****************
    # k1
    k1Bs  =  FB(Bs,Ps,P2s,Ba,g,K,η,j,boolean_B,IΔx)   # k1 for B
    k1Ps  =  FP(Bs,Ps,Pa,β,δs,η,j,boolean_P,IΔx)   # k1 for P
    k1P2s =  FP(Bs,P2s,P2a,β,δs2,η,j,boolean_P2,IΔx)

    k1Ba  =  FBA(Ba,Bs,δa,j,boolean_B)   # k1 for B
    k1Pa  =  FPA(Pa,Ps,δa,j,boolean_P)   # k1 for P
    k1P2a =  FPA(P2a,P2s,δa,j,boolean_P2)
    #==================#
    # k2
    k2Bs  =  FB(Bs+0.5*k1Bs*Δt,Ps+0.5*k1Ps*Δt,P2s+0.5*k1P2s*Δt,Ba+0.5*k1Ba*Δt,g,K,η,j,boolean_B,IΔx)
    k2Ps  =  FP(Bs+0.5*k1Bs*Δt,Ps+0.5*k1Ps*Δt,Pa+0.5*k1Pa*Δt,β,δs,η,j,boolean_P,IΔx)
    k2P2s =  FP(Bs+0.5*k1Bs*Δt,P2s+0.5*k1P2s*Δt,P2a+0.5*k1P2a*Δt,β,δs2,η,j,boolean_P2,IΔx)

    k2Ba  =  FBA(Ba+0.5*k1Ba*Δt,Bs+0.5*k1Bs*Δt,δa,j,boolean_B)
    k2Pa  =  FPA(Pa+0.5*k1Pa*Δt,Ps+0.5*k1Ps*Δt,δa,j,boolean_P)
    k2P2a =  FPA(P2a+0.5*k1P2a*Δt,P2s+0.5*k1P2s*Δt,δa,j,boolean_P2)
    #==================#
    # k3
    k3Bs  =  FB(Bs+0.5*k2Bs*Δt,Ps+0.5*k2Ps*Δt,P2s+0.5*k2P2s*Δt,Ba+0.5*k2Ba*Δt,g,K,η,j,boolean_B,IΔx)
    k3Ps  =  FP(Bs+0.5*k2Bs*Δt,Ps+0.5*k2Ps*Δt,Pa+0.5*k2Pa*Δt,β,δs,η,j,boolean_P,IΔx)
    k3P2s =  FP(Bs+0.5*k2Bs*Δt,P2s+0.5*k2P2s*Δt,P2a+0.5*k2P2a*Δt,β,δs2,η,j,boolean_P2,IΔx)

    k3Ba  =  FBA(Ba+0.5*k2Ba*Δt,Bs+0.5*k2Bs*Δt,δa,j,boolean_B)
    k3Pa  =  FPA(Pa+0.5*k2Pa*Δt,Ps+0.5*k2Ps*Δt,δa,j,boolean_P)
    k3P2a =  FPA(P2a+0.5*k2P2a*Δt,P2s+0.5*k2P2s*Δt,δa,j,boolean_P2)
    #==================#
    # k4
    k4Bs  =  FB(Bs+k3Bs*Δt,Ps+k3Ps*Δt,P2s+k3P2s*Δt,Ba+k3Ba*Δt,g,K,η,j,boolean_B,IΔx)
    k4Ps  =  FP(Bs+k3Bs*Δt,Ps+k3Ps*Δt,Pa+k3Pa*Δt,β,δs,η,j,boolean_P,IΔx)
    k4P2s =  FP(Bs+k3Bs*Δt,P2s+k3P2s*Δt,P2a+k3P2a*Δt,β,δs2,η,j,boolean_P2,IΔx)

    k4Ba  =  FBA(Ba+k3Ba*Δt,Bs+k3Bs*Δt,δa,j,boolean_B)
    k4Pa  =  FPA(Pa+k3Pa*Δt,Ps+k3Ps*Δt,δa,j,boolean_P)
    k4P2a =  FPA(P2a+k3P2a*Δt,P2s+k3P2s*Δt,δa,j,boolean_P2)
    
    return 1/6*(k1Bs+2*k2Bs+2*k3Bs+k4Bs)*Δt, 1/6*(k1Ba+2*k2Ba+2*k3Ba+k4Ba)*Δt, 1/6*(k1Ps+2*k2Ps+2*k3Ps+k4Ps)*Δt , 1/6*(k1Pa+2*k2Pa+2*k3Pa+k4Pa)*Δt, 1/6*(k1P2s+2*k2P2s+2*k3P2s+k4P2s)*Δt , 1/6*(k1P2a+2*k2P2a+2*k3P2a+k4P2a)*Δt



@numba.njit
def lax_wendroff_periodic(M,b,c,ϕ=1):
    
    M_left   = np.roll(M,1)
    M_right  = np.roll(M,-1)

    # Lax-Wendroff with flux limiter (to avoid spurious oscillations)
    
    return c*(1-0.5*ϕ*(1-c))*M_left + (1+c*ϕ)*(1-c)*M  -0.5*c*(1-c)*ϕ*M_right

@numba.njit
def lax_wendroff_open(M,v,c,ϕ=1):
    
    if v > 0:
        
        M_left    = np.roll(M,1)
        M_left[0] = 0
        return M_left
    
    else:
        
        M_right   = np.roll(M,-1)
        M_right[-1] = 0
        return M_right
    
  #  Lax-Wendroff with flux limiter (not needed if C = 1)
  #  return c*(1-0.5*ϕ*(1-c))*M_left + (1+c*ϕ)*(1-c)*M  -0.5*c*(1-c)*ϕ*M_right

@numba.njit                   
def fluxes(fs,fa,j,boolean):
    # fs (fa) surface (atmosphere) density
    # flux sign --> Down = positive

    return -j*(fs-fa)*boolean

@numba.njit
def laplacian_open(M):
    
    L_left   = np.roll(M,1)
    L_right  = np.roll(M,-1)
    
    L_left[0]   = M[0]     # No concentration gradient on the edges (there is a fake point with the same concentration)
    L_right[-1] = M[-1]
    
    L = L_left + L_right - 2*M
    
    return L

#============================#
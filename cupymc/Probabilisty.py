# User instruction

# 1. Author: Luna (Yuxuan Mao)

# 2. Date: from 11/09 to ...

# 3. Function: class of Probabilistically 
                # determination of the scattered direction
#    including:
#    3.1 

# 4. Copyright
# 4.1 Optical Metrology & Image Processing Laboratory (MIP-OptiLab)
# Sincere and hearty thanks to: Prof.Juan Campos, Prof.Angel Lizana, Irene Estévez
# Extremely respect to the author of Matlab version: Albert Van Eeckhout Alsinet
# really grateful to Ivan Montes
# LOVE YOU!!!!!

# Muchas gracias, Barcelona, y mi universidad, Universitat Autònoma de Barcelona.
# A mi me gusta esta ciudad.

# 4.2 CSC funding
# Supported by the China Scholarship Council
# It is my honor to benefit from my motherland

import math
from scipy.special import lpmn
# Sequence of associated Legendre functions of the first kind
# import numpy as np
import cupy as np

class Sphere_probabilisty:
    
    def __init__(self,nalfa,nbeta,n,an,bn,nmax):
        self.nalfa=nalfa
        self.nbeta=nbeta
        self.n=n
        self.an=an
        self.bn=bn
        self.nmax=nmax

    def cal_pin_taon(self,n,theta):
        # 5.1.1 function: 
        # Target: for associated Legendre function Pn_m
        #  
        # Input: n: maximum of length
        #        theta: each angles 
        # output:
        #        Angle-Dependent Functions (recurrence from the relations)
        #        pi_n:
        #        tao_n:

        # 5.1.2 Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 90 -- eq.4.25
        # page 105 -- 4.3.1 Angle-Dependent Functions 
        
        # 5.1.3 note for associated Legendre functions
        # scipy.special.lpmn(m, n, z)
        # Parameters: m int
        #                 |m| <= n; the order of the Legendre function.
        #             n int
        #                 where n >= 0; the degree of the Legendre function. Often called l (lower case L) in descriptions of the associated Legendre function
        #             z float
        #                 Input value.
        # Returns:  Pmn_z: (m+1, n+1) array
        #             Values for all orders 0..m and degrees 0..n
        #           Pmn_d_z(m+1, n+1) array
        #             Derivatives for all orders 0..m and degrees 0..n
        # Note: The angle-dependent functions appear to pose no particular computational problem
        # TODO 

        pin=[0]*(n+1)
        # pin[0]=1
        pin[1]= 1 
        # 2*math.cos(theta)
        
        taon=[0]*(n+1)
        taon[1]=1*math.cos(theta)*pin[1]-(1+1)*pin[0]

        for i in range(2,n+1):
            pin[i]=(2*i-1)*math.cos(theta)*pin[i-1]/(i-1)-i*pin[i-2]/(i-1)
            # print(pin[i])
            taon[i]=i*math.cos(theta)*pin[i]-(i+1)*pin[i-1]
        
        del pin[0]
        del taon[0]
        return pin,taon
    
    def cal_pin_taon2(self,n,theta):
        pass

    def cal_S1_S2(self,theta):
        # 5.1.2.1 function: 
        # Target: for computing S1 S2 in scattered field 
        
        # Input: theta for each angles
        # output: S1,S2 parameters in scattered electric field 

        # 5.1.2.2 Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 112 -- eq.4.74

        pn,tn=self.cal_pin_taon(self.nmax,theta)
        pn = np.array(pn)
        tn = np.array(tn)

        # S1=np.zeros([1,self.nmax])
        # S2=np.zeros([1,self.nmax])
        
        # for n in range(0,self.nmax):
        n=np.arange(1,self.nmax,self.nmax)
        # n=list(map(float,n))
        s1=(2*n+1)*(self.an*pn+self.bn*tn)/(n*(n+1))
        s2=(2*n+1)*(self.an*tn+self.bn*pn)/(n*(n+1))
        
        S1=np.sum(s1)
        S2=np.sum(s2)

        return S1,S2

    def Scattering_matrix(self):
        # 5.1.3.1 function: 
        # Target: for computing a matrix of Mie coefficients for s11,s12,s33,s34
        
        # Input: self
        # output: Scattering_matrix 4*n_alfa

        # 5.1.3.2 Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 111-114 -- 4.4.4 Scattering Matrix 
        Scattering_matrix= np.zeros((4,self.nalfa),dtype=np.complex_)

        # !!!!!!!!!
        # it is where alfa has been created
        for ii in range(1,self.nalfa):
            alfa=(ii-1)*math.pi/(self.nalfa-1)
            S1,S2 = self.cal_S1_S2(alfa)
            
            S1=np.array(S1)
            S2=np.array(S2)

            s11=(abs(S2)**2+abs(S1)**2)/2
            s12=(abs(S2)**2-abs(S1)**2)/2

            s33=(np.conjugate(S2)*S1+np.conjugate(S1)*S2)/2
            s34_a=(np.conjugate(S2)*S1-np.conjugate(S1)*S2)/2
            real_s34=-np.imag(s34_a)
            imag_s34=np.real(s34_a)
            s34=complex(real_s34,imag_s34)
            
            Scattering_matrix[0,ii]=s11
            Scattering_matrix[1,ii]=s12
            Scattering_matrix[2,ii]=s33
            Scattering_matrix[3,ii]=s34

        return Scattering_matrix

    def cal_norm_P_alfa(self,Scattering_matrix):
        # 5.1.4.1 instruction
        # Probability to be scattered in a particular direction (alfa,beta) 
        # integrate the probability, over all the possible beta angles
        # to get P_alfa
        # source: albert's code
        # and his "Formules format tesis"

        # 5.1.4.2 Calculate the P(alfa)
        # normalized probabilities 
        # if we ingore the beta, then
        # P_alfa = S11(alfa)
        P_norm= np.cumsum(Scattering_matrix[0,:])
        P_alfa_norm=P_norm/P_norm[self.nalfa-1]

        return P_alfa_norm
    
    def cal_sca_angle(self,Scattering_matrix,stokes):
        
        Palf_norm = self.cal_norm_P_alfa(Scattering_matrix)

        s11 = Scattering_matrix[0,:]
        s12 = Scattering_matrix[1,:]
        s33 = Scattering_matrix[2,:]
        s34 = Scattering_matrix[3,:]
        Ii  = stokes[0]
        Qi  = stokes[1]
        Ui  = stokes[2]
        Vi  = stokes[3]
        
        # 6.2.1 
        # α value is determined using a pseudorandom number
        # it is based on the difference minimization 
        # source: Albert's tesis
        
        # 6.2.2.1 get a alfa_n randomly 
        # eq 1-29
        list_alfa=abs(np.random.rand(1)-Palf_norm)
        alfa_min=min(list_alfa)
        alfa_n=np.array(list_alfa).argmin()   
        
        # 6.2.2.2 create beta
        # for n2 in range(1,self.nbeta):
            # beta = (n2-1)*2*math.pi/(self.nbeta-1)

                
        beta = [(n2-1)*2*math.pi/(self.nbeta-1) for n2 in range(1,self.nbeta)]
        beta = np.array(beta)

        # 6.2.2.3 calculate the P(alfa,beta)
        P=s11[alfa_n]*Ii+s12[alfa_n]*(Qi*np.cos(2*beta)+Ui*np.sin(2*beta))
        P_cum=np.cumsum(P)
        P_cum=P_cum/P_cum[self.nbeta-2]

        # 6.2.2.4 get a beta_n randomly 
        # Choose random alfa and beta teking into account the 
        # probability distribution P.
        
        list_beta=abs(np.random.rand(1)-P_cum)
        beta_min=min(list_beta)
        beta_n=np.array(list_beta).argmin()  

        alfa=(alfa_n-1)*math.pi/(self.nalfa-1)
        beta=(beta_n-1)*2*math.pi/(self.nbeta-1)


        # 6.2.2 
        # calculate the I0, Qb, Ub, Vo
        I0=s11[alfa_n]*Ii+s12[alfa_n]*(Qi*math.cos(2*beta)+Ui*math.sin(2*beta))
        Qb=s12[alfa_n]*Ii+s11[alfa_n]*(Qi*math.cos(2*beta)+Ui*math.sin(2*beta))
        Ub=s34[alfa_n]*Vi+s33[alfa_n]*(Ui*math.cos(2*beta)-Qi*math.sin(2*beta))
        Vo=s33[alfa_n]*Vi+s34[alfa_n]*(Qi*math.sin(2*beta)-Ui*math.cos(2*beta))

        stokes=np.array([I0, Qb, Ub, Vo])

        return alfa,beta,stokes
    
class cylinder_prosibility():

    def __init__(self,an1,an2,bn1,bn2,nzeta,nphi):
        self.an1 =an1
        self.an2 =an2
        self.bn1 =bn1
        self.bn2 =bn2
        self.nzeta=nzeta
        self.nphi=nphi
        # self.n=n

    def cylinder_cal_T(self,phi):
        # phi=[ii*2*math.pi/(self.nphi-1) for ii in range(0,self.nphi)]
        Bigtheta = np.pi - phi
        T1 = np.sum(2*self.bn1*np.cos(self.n*Bigtheta))-self.bn1[0]
        T2 = np.sum(2*self.an2*np.cos(self.n*Bigtheta))-self.an1[0]
        T3 = np.sum(-2j*self.an1*np.sin(self.n*Bigtheta))
        T4 = np.sum(-2j*self.bn2*np.sin(self.n*Bigtheta))

        return T1, T2, T3, T4
        
            # T11=(np.abs(T1)**2+np.abs(T2)**2)/2
            # T12=(np.abs(T1)**2-np.abs(T2)**2)/2
            # T33=np.real(T1*np.conjugate(T2))
            # T34=np.imag(T1*np.conjugate(T2))
    
    def cal_norm_P_zeta(self,P):
        # n_cylinder_theta = 360
        P_norm= np.cumsum(P[0,:])
        P_zeta_norm=P_norm/P_norm[-1]

        return P_zeta_norm
    
    def cal_sca_angle(self,P,stokes,d_vectors):
        
        Palf_norm = self.cal_norm_P_alfa(P)

        # s11 = Scattering_matrix[0,:]
        # s12 = Scattering_matrix[1,:]
        # s33 = Scattering_matrix[2,:]
        # s34 = Scattering_matrix[3,:]
        Ii  = stokes[0]
        Qi  = stokes[1]
        Ui  = stokes[2]
        Vi  = stokes[3]
        
        # 6.2.1 
        # α value is determined using a pseudorandom number
        # it is based on the difference minimization 
        # source: Albert's tesis
        
        # 6.2.2.1 get a alfa_n randomly 
        # eq 1-29
        list_theta=abs(np.random.rand(1)-Palf_norm)
        theta_min=min(list_theta)
        theta_n=np.array(list_theta).argmin()   
        
        # 6.2.2.2 create beta
        # for n2 in range(1,self.nbeta):
            # beta = (n2-1)*2*math.pi/(self.nbeta-1)

        self.nphi = 360
                
        phi = [(n2)*2*math.pi/(self.nphi) for n2 in range(1,self.nphi)]
        phi = np.array(phi)

        # 6.2.2.3 calculate the P(theta,phi)
        # theta is out-angle
        # where phi is rotating angle
        # for i in phi:
        p_phi = np.zeros((self.nphi,1))
        for id, phi_angle in enumerate(phi):
            p_phi[id] = P[:,id]
            pp = p_phi[id]
        
        P_cum=np.cumsum(pp)
        P_cum=P_cum/P_cum[self.nphi-1]


        # 6.2.2.4 get a beta_n randomly 
        # Choose random alfa and beta teking into account the 
        # probability distribution P.
        
        list_phi=abs(np.random.rand(1)-P_cum)
        phi_min=min(list_phi)
        phi_n=np.array(list_phi).argmin()  

        theta=(theta_n-1)*math.pi/(self.ntheta-1)
        phi=(phi_n-1)*2*math.pi/(self.nphi-1)


        # 6.2.2 
        # calculate the I0, Qb, Ub, Vo

        T1, T2, T3, T4 = self.cylinder_cal_T(phi)
        #  TODO 
        # check if it is true
        #  just based on mermory

        M = self.cylinder_scattering_matrix(T1[theta_n],T2[theta_n],T3[theta_n],T4[theta_n])
        
        # rotation for cylinder scattering

        l =np.array([1,0,0])
        vi = np.cross(l,d_vectors)
        angle_difference = np.arccos(np.dot(d_vectors,vi)/(np.linalg.norm(d_vectors)*np.linalg.norm(vi)))
        R_matrix = np.array([1,0,0,0],
                            [0,np.cos(2*angle_difference),np.sin(2*angle_difference),0],
                            [0,-np.sin(2*angle_difference),np.cos(2*angle_difference),0],
                            [0,0,0,1]
                            )

        Sout = M*R_matrix*stokes
        
        # stokes=np.array([I0, Qb, Ub, Vo])

        return theta, phi, Sout
    
    def cylinder_scattering_matrix(self, T1,T2,T3,T4):
        m11= (np.abs(T1)**2+np.abs(T2)**2+np.abs(T3)**2+np.abs(T4)**2)/2
        m12= (np.abs(T1)**2-np.abs(T2)**2+np.abs(T3)**2-np.abs(T4)**2)/2
        m13=np.real(T1*np.conjugate(T4)+T2*np.conjugate(T3))
        m14=np.imag(T1*np.conjugate(T4)-T2*np.conjugate(T3))
        m21= (np.abs(T1)**2-np.abs(T2)**2-np.abs(T3)**2+np.abs(T4)**2)/2
        m22= (np.abs(T1)**2+np.abs(T2)**2-np.abs(T3)**2-np.abs(T4)**2)/2
        m23=np.real(T1*np.conjugate(T4)-T2*np.conjugate(T3))
        m24=np.imag(T1*np.conjugate(T4)+T2*np.conjugate(T3))

        m31=np.real(T1*np.conjugate(T3)+T2*np.conjugate(T4))
        m32=np.real(T1*np.conjugate(T3)-T2*np.conjugate(T4))
        m33=np.real(np.conjugate(T1)*T2+np.conjugate(T3)*T4)
        m34=np.imag(T1*np.conjugate(T2)+T3*np.conjugate(T4))

        m41=np.imag(np.conjugate(T1)*T3+T2*np.conjugate(T4))
        m42=np.imag(np.conjugate(T1)*T3-T2*np.conjugate(T4))
        m43=np.imag(np.conjugate(T1)*T2-np.conjugate(T3)*T4)
        m44=np.real(np.conjugate(T1)*T2-np.conjugate(T3)*T4)

        M = np.array([m11,m12,m13,m14], [m21,m22,m23,m24], [m31,m32,m33,m34], [m41,m42,m43,m44])
        return M
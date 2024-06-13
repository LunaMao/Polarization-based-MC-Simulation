# User instruction

# 1. Author: Luna (Yuxuan Mao)

# 2. Date: from 11/10 to ...

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
import numpy as np
from Probabilisty import Sphere_probabilisty
from operator import methodcaller
from multiprocessing import Pool, TimeoutError
import time
from scipy.special import hermite,eval_hermite
from utility import mymath,measurement
from scipy.special import genlaguerre
from scipy.special import hyp2f1
# import treams
from tqdm import tqdm
from alive_progress import alive_bar
import torch
import torch.multiprocessing as mp
CUDA_LAUNCH_BLOCKING="1"
# TORCH_USE_CUDA_DSA
# from tqdm.auto import tqdm
# 1. Characterize the source photon by choosing its position, 
# direction, and initial polarization.

# 2. Define the site and the type of next event 
# (scattering on particle, transmission, or reflection on the interfaces). 
# If the photon has reached the boundaries of the domain of simulations, go to step 5.


# 3. Change the direction of propagation of the photon and its polarization 
# according to the type of event.

# 4. Scoring, check if the photon stays within the domain of simulations. 
# If so, back to step 2.

# 5. The photon path terminates whenever it reaches the boundaries 
# of the domain of simulations which means that it has no return possibility.


class photons_event:

    def __init__(self,Scattering_matrix,stokes,Palf_norm,nalfa,nbeta,batch_size):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # print("using GPU:",torch.cuda.get_device_name(0))
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device =torch.device('cpu')
            self.dtype = torch.FloatTensor
            # print("Using CPU")

        if torch.is_tensor(Scattering_matrix):

            if Palf_norm.size()[0]==batch_size:
                self.Scattering_matrix=Scattering_matrix
                self.stokes = stokes
                self.s11 =Scattering_matrix[:,0,:]
                self.s12 = Scattering_matrix[:,1,:]
                self.s33 = Scattering_matrix[:,2,:]
                self.s34 = Scattering_matrix[:,3,:]
                self.Ii  = stokes[:,0]
                self.Qi  = stokes[:,1]
                self.Ui  = stokes[:,2]
                self.Vi  = stokes[:,3]
                self.Palf_norm = Palf_norm
                # self.palf_norm = Palf_norm
                # the default number we set for alfa and beta 
                self.nalfa = nalfa
                self.nbeta = nbeta
            else:
                self.Scattering_matrix=Scattering_matrix.repeat(batch_size,1,1)
                self.stokes = stokes.repeat(batch_size,1)
                self.s11 =Scattering_matrix[:,0,:]
                self.s12 = Scattering_matrix[:,1,:]
                self.s33 = Scattering_matrix[:,2,:]
                self.s34 = Scattering_matrix[:,3,:]
                self.Ii  = stokes[:,0]
                self.Qi  = stokes[:,1]
                self.Ui  = stokes[:,2]
                self.Vi  = stokes[:,3]
                self.Palf_norm = Palf_norm.repeat(batch_size,1)
                # self.palf_norm = Palf_norm
                # the default number we set for alfa and beta 
                self.nalfa = nalfa
                self.nbeta = nbeta
        else:
            self.Scattering_matrix=torch.tensor(Scattering_matrix).type(self.dtype).repeat(batch_size,1,1)
            # bacth size * 4 *nalfa
            self.stokes = torch.tensor(stokes).type(self.dtype).repeat(batch_size,1)
            # bacth size * 4

            self.s11 = self.Scattering_matrix[:,0,:]
            self.s12 = self.Scattering_matrix[:,1,:]
            self.s33 = self.Scattering_matrix[:,2,:]
            self.s34 = self.Scattering_matrix[:,3,:]
            self.Ii  = self.stokes[:,0]
            self.Qi  = self.stokes[:,1]
            self.Ui  = self.stokes[:,2]
            self.Vi  = self.stokes[:,3]

            # bacth size * nalfa
            self.Palf_norm = torch.tensor(Palf_norm).type(self.dtype).repeat(batch_size,1)
            self.palf_norm = torch.tensor(Palf_norm).type(self.dtype)

            # the default number we set for alfa and beta 
            self.nalfa = torch.tensor(nalfa).type(self.dtype)
            self.nbeta = torch.tensor(nbeta).type(self.dtype)
        
    def mie_scattering_particle(self):

        # 6.2.1 
        # α value is determined using a pseudorandom number
        # it is based on the difference minimization 
        # source: Albert's tesis
        
        # 6.2.2.1 get a alfa_n randomly 
        # eq 1-29
        list_alfa=abs(torch.rand(1).type(self.dtype)-self.Palf_norm)
        alfa_min=min(list_alfa)
        alfa_n=torch.argmin(list_alfa)
        
        # 6.2.2.2 create beta
        # for n2 in range(1,self.nbeta):
            # beta = (n2-1)*2*math.pi/(self.nbeta-1)

                
        beta = [(n2-1)*2*math.pi/(self.nbeta-1) for n2 in torch.arange(1,self.nbeta)]
        beta = torch.tensor(beta).type(self.dtype)
        # 6.2.2.3 calculate the P(alfa,beta)
        P=self.s11[alfa_n]*self.Ii+self.s12[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
        P_cum=torch.cumsum(P,dim=0)
        P_cum=P_cum/P_cum[int(self.nbeta.item())-2]

        # 6.2.2.4 get a beta_n randomly 
        # Choose random alfa and beta teking into account the 
        # probability distribution P.
        
        list_beta=abs(torch.rand(1).type(self.dtype)-P_cum)
        beta_min=min(list_beta)
        beta_n=torch.argmin(list_beta)

        alfa=(alfa_n-1)*torch.pi/(self.nalfa-1)
        beta=(beta_n-1)*2*torch.pi/(self.nbeta-1)


        # 6.2.2 
        # calculate the I0, Qb, Ub, Vo
        I0=self.s11[alfa_n]*self.Ii+self.s12[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
        Qb=self.s12[alfa_n]*self.Ii+self.s11[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
        Ub=self.s34[alfa_n]*self.Vi+self.s33[alfa_n]*(self.Ui*torch.cos(2*beta)-self.Qi*torch.sin(2*beta))
        Vo=self.s33[alfa_n]*self.Vi+self.s34[alfa_n]*(self.Qi*torch.sin(2*beta)-self.Ui*torch.cos(2*beta))

        stokes=torch.tensor([I0, Qb, Ub, Vo]).type(self.dtype)

        return alfa,beta,stokes

    def mie_scattering_particles(self,batch_size,alive_mask):

        # 6.2.1 
        # α value is determined using a pseudorandom number
        # it is based on the difference minimization 
        # source: Albert's tesis
        
        # 6.2.2.1 get a alfa_n randomly 
        # eq 1-29
        # Palf_norm = self.Palf_norm[alive_mask.squeeze(dim=1),:]
        list_alfa=abs(torch.rand_like(self.Palf_norm).type(self.dtype)-self.Palf_norm)
        alfa_min,alfa_n =list_alfa.min(dim=1,keepdim=True)
   
        # 6.2.2.2 create beta
        # for n2 in range(1,self.nbeta):
            # beta = (n2-1)*2*math.pi/(self.nbeta-1)

        beta = [(n2-1)*2*math.pi/(self.nbeta-1) for n2 in torch.arange(1,self.nbeta)]
        
        beta = torch.tensor(beta).type(self.dtype)
        # beta_T = torch.transpose(beta)

        # 6.2.2.3 calculate the P(:,beta)
        
        # Term4=torch.transpose(self.Ui)*torch.sin(2*beta)
        # Term3=torch.transpose(self.Qi)*torch.cos(2*beta)
        P=torch.zeros(batch_size,int(self.nbeta.item())-1).type(self.dtype)
        P_cum=torch.zeros(batch_size,int(self.nbeta.item())-1).type(self.dtype)
        P_cum2 =torch.zeros(batch_size,int(self.nbeta.item())-1).type(self.dtype)
        for i in range(0,batch_size):
            # batch * nbeta
            P[i,:]=self.s11[i,alfa_n[i]]*self.Ii[i]+self.s12[i,alfa_n[i]]*(self.Qi[i]*torch.cos(2*beta)+self.Ui[i]*torch.sin(2*beta))
            bridge = P[i,:]
            P_cum[i,:]=torch.cumsum(bridge,dim=0)
            P_cum2[i,:]=P_cum[i,:]/P_cum[i,int(self.nbeta.item())-2]

        # 6.2.2.4 get a beta_n randomly 
        # Choose random alfa and beta taking into account the 
        # probability distribution P.
        
        list_beta=abs(torch.rand_like(P_cum2).type(self.dtype).type(self.dtype)-P_cum2)
        beta_min,beta_n =list_beta.min(dim=1,keepdim=True)
        # beta_min=min(list_beta)
        # beta_n=torch.argmin(list_beta)

        alfa=(alfa_n-1)*torch.pi/(self.nalfa-1)

        # alfa = alfa
        print("the size of alfa is {}".format(alfa.size()))
        # print(alfa_n.size())
        beta=(beta_n-1)*2*torch.pi/(self.nbeta-1)


        # 6.2.2 
        # calculate the I0, Qb, Ub, Vo 
        stokes=torch.zeros(batch_size,4).type(self.dtype)
        # for batch of photons 


        for j in range(0,batch_size):
            # I0
            stokes[j,0]=self.s11[i,alfa_n[i]]*self.Ii[i]+self.s12[i,alfa_n[i]]*(self.Qi[i]*torch.cos(2*beta[i])+self.Ui[i]*torch.sin(2*beta[i]))
            # Qb
            stokes[j,1]=self.s12[i,alfa_n[i]]*self.Ii[i]+self.s11[i,alfa_n[i]]*(self.Qi[i]*torch.cos(2*beta[i])+self.Ui[i]*torch.sin(2*beta[i]))
            # Ub
            stokes[j,2]=self.s34[i,alfa_n[i]]*self.Vi[i]+self.s33[i,alfa_n[i]]*(self.Ui[i]*torch.cos(2*beta[i])-self.Qi[i]*torch.sin(2*beta[i]))
            # Vo
            stokes[j,3]=self.s33[i,alfa_n[i]]*self.Vi[i]+self.s34[i,alfa_n[i]]*(self.Qi[i]*torch.sin(2*beta[i])-self.Ui[i]*torch.cos(2*beta[i]))

        return alfa,beta,stokes
# class photon_event:
#     # 6.1.1 Name 
#     # 6.1.2 functions: 
#     # Computes the output direction, Vxo,Vyo,Vzo 
#     # based on the photon events

#     def __init__(self,Scattering_matrix,stokes,Palf_norm,nalfa,nbeta):

#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#             # print("using GPU:",torch.cuda.get_device_name(0))
#             self.dtype = torch.cuda.FloatTensor
#         else:
#             self.device =torch.device('cpu')
#             self.dtype = torch.FloatTensor
#             # print("Using CPU")

#         if torch.is_tensor(Scattering_matrix):
#             self.Scattering_matrix=Scattering_matrix
#             self.stokes = stokes.to(self.device)
#             self.s11 =Scattering_matrix[0,:]
#             self.s12 = Scattering_matrix[1,:]
#             self.s33 = Scattering_matrix[2,:]
#             self.s34 = Scattering_matrix[3,:]
#             self.Ii  = stokes[0]
#             self.Qi  = stokes[1]
#             self.Ui  = stokes[2]
#             self.Vi  = stokes[3]
#             self.Palf_norm = Palf_norm
#             # the default number we set for alfa and beta 
#             self.nalfa = nalfa
#             self.nbeta = nbeta
#         else:
#             self.Scattering_matrix=torch.tensor(Scattering_matrix).type(self.dtype)
#             self.stokes = torch.tensor(stokes).type(self.dtype)

#             self.s11 = self.Scattering_matrix[0,:]
#             self.s12 = self.Scattering_matrix[1,:]
#             self.s33 = self.Scattering_matrix[2,:]
#             self.s34 = self.Scattering_matrix[3,:]
#             self.Ii  = self.stokes[0]
#             self.Qi  = self.stokes[1]
#             self.Ui  = self.stokes[2]
#             self.Vi  = self.stokes[3]
        
#             self.Palf_norm = torch.tensor(Palf_norm).type(self.dtype)
#             # the default number we set for alfa and beta 
#             self.nalfa = torch.tensor(nalfa).type(self.dtype)
#             self.nbeta = torch.tensor(nbeta).type(self.dtype)
        


#     def mie_scattering_particle(self):

#         # 6.2.1 
#         # α value is determined using a pseudorandom number
#         # it is based on the difference minimization 
#         # source: Albert's tesis
        
#         # 6.2.2.1 get a alfa_n randomly 
#         # eq 1-29
#         list_alfa=abs(torch.rand(1).type(self.dtype)-self.Palf_norm)
#         alfa_min=min(list_alfa)
#         alfa_n=torch.argmin(list_alfa)
        
#         # 6.2.2.2 create beta
#         # for n2 in range(1,self.nbeta):
#             # beta = (n2-1)*2*math.pi/(self.nbeta-1)

                
#         beta = [(n2-1)*2*math.pi/(self.nbeta-1) for n2 in torch.arange(1,self.nbeta)]
#         beta = torch.tensor(beta).type(self.dtype)
#         # 6.2.2.3 calculate the P(alfa,beta)
#         P=self.s11[alfa_n]*self.Ii+self.s12[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
#         P_cum=torch.cumsum(P,dim=0)
#         P_cum=P_cum/P_cum[int(self.nbeta.item())-2]

#         # 6.2.2.4 get a beta_n randomly 
#         # Choose random alfa and beta teking into account the 
#         # probability distribution P.
        
#         list_beta=abs(torch.rand(1).type(self.dtype)-P_cum)
#         beta_min=min(list_beta)
#         beta_n=torch.argmin(list_beta)

#         alfa=(alfa_n-1)*torch.pi/(self.nalfa-1)
#         beta=(beta_n-1)*2*torch.pi/(self.nbeta-1)


#         # 6.2.2 
#         # calculate the I0, Qb, Ub, Vo
#         I0=self.s11[alfa_n]*self.Ii+self.s12[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
#         Qb=self.s12[alfa_n]*self.Ii+self.s11[alfa_n]*(self.Qi*torch.cos(2*beta)+self.Ui*torch.sin(2*beta))
#         Ub=self.s34[alfa_n]*self.Vi+self.s33[alfa_n]*(self.Ui*torch.cos(2*beta)-self.Qi*torch.sin(2*beta))
#         Vo=self.s33[alfa_n]*self.Vi+self.s34[alfa_n]*(self.Qi*torch.sin(2*beta)-self.Ui*torch.cos(2*beta))

#         stokes=torch.tensor([I0, Qb, Ub, Vo]).type(self.dtype)

#         return alfa,beta,stokes

#     def mie_cylinder_cal_P_zeta(self,P,nzeta):
#         # n_cylinder_theta = 360
#         p_zeta = np.zeros((nzeta,1))
#         for id in range(0,nzeta):
#             p_zeta[id] = np.sum(P[id,:])
#             # pp = p_phi[id]
#         P_norm= np.cumsum(p_zeta)
#         P_zeta_norm=P_norm/P_norm[nzeta-1]

#         return P_zeta_norm
    
#     def mie_cylinder_cal_T(self,phi,an1,an2,bn1,bn2,id_zeta):
#         # phi=[ii*2*math.pi/(self.nphi-1) for ii in range(0,self.nphi)]
#         Bigtheta = np.pi - phi
#         id_zeta = int(id_zeta)
#         N =an1.shape[1]
#         T1=0
#         T2=0
#         T3=0
#         T4=0
#         # TODO maybe bug here!!!
#         for id_n in range(0,N):
#             T1 = T1 + 2*bn1[id_zeta,id_n]*np.cos(id_n*Bigtheta)
#             T2 = T2 + 2*an2[id_zeta,id_n]*np.cos(id_n*Bigtheta)
#             T3 = T3 - 2j*an1[id_zeta,id_n]*np.sin(id_n*Bigtheta)
#             T4 = T4 - 2j*bn2[id_zeta,id_n]*np.sin(id_n*Bigtheta)

#             # T2 = np.sum(2*an2[id_zeta,:]*np.cos(id_n*Bigtheta))-an1[0]
#             # T3 = np.sum(-2j*an1[id_zeta,:]*np.sin(id_n*Bigtheta))
#             # T4 = np.sum(-2j*bn2[id_zeta,:]*np.sin(id_n*Bigtheta))

#         return T1-bn1[id_zeta,0], T2-an2[id_zeta,0], T3, T4

#     def mie_cylinder_Smatrix(self, T1,T2,T3,T4):
#         m11= (np.abs(T1)**2+np.abs(T2)**2+np.abs(T3)**2+np.abs(T4)**2)/2
#         m12= (np.abs(T1)**2-np.abs(T2)**2+np.abs(T3)**2-np.abs(T4)**2)/2
#         m13=np.real(T1*np.conjugate(T4)+T2*np.conjugate(T3))
#         m14=np.imag(T1*np.conjugate(T4)-T2*np.conjugate(T3))
#         m21= (np.abs(T1)**2-np.abs(T2)**2-np.abs(T3)**2+np.abs(T4)**2)/2
#         m22= (np.abs(T1)**2+np.abs(T2)**2-np.abs(T3)**2-np.abs(T4)**2)/2
#         m23=np.real(T1*np.conjugate(T4)-T2*np.conjugate(T3))
#         m24=np.imag(T1*np.conjugate(T4)+T2*np.conjugate(T3))

#         m31=np.real(T1*np.conjugate(T3)+T2*np.conjugate(T4))
#         m32=np.real(T1*np.conjugate(T3)-T2*np.conjugate(T4))
#         m33=np.real(np.conjugate(T1)*T2+np.conjugate(T3)*T4)
#         m34=np.imag(T1*np.conjugate(T2)+T3*np.conjugate(T4))

#         m41=np.imag(np.conjugate(T1)*T3+T2*np.conjugate(T4))
#         m42=np.imag(np.conjugate(T1)*T3-T2*np.conjugate(T4))
#         m43=np.imag(np.conjugate(T1)*T2-np.conjugate(T3)*T4)
#         m44=np.real(np.conjugate(T1)*T2-np.conjugate(T3)*T4)

#         M = [m11,m12,m13,m14,m21,m22,m23,m24,m31,m32,m33,m34,m41,m42,m43,m44]
#         Marray=np.array(M)
#         mm = Marray.reshape(4,4)
#         # M = np.array([m11,m12,m13,m14]),
#         #              np.array([m21,m22,m23,m24]),
#         #              np.array([m31,m32,m33,m34]),
#         #              np.array([m41,m42,m43,m44]))
#         return mm
    
#     def mie_scattering_cylinder(self,P,d_vectors,l,n_theta,n_phi,an1,an2,bn1,bn2):
        
#         # P phase function
#         # M ??
#         # default: l =np.array([1,0,0])
#         # l =np.array([1,0,0])
#         # n id_zeta
        
#         Palf_norm = self.mie_cylinder_cal_P_zeta(P,n_theta)

#         # s11 = Scattering_matrix[0,:]
#         # s12 = Scattering_matrix[1,:]
#         # s33 = Scattering_matrix[2,:]
#         # s34 = Scattering_matrix[3,:]
#         # Ii  = stokes[0]
#         # Qi  = stokes[1]
#         # Ui  = stokes[2]
#         # Vi  = stokes[3]
        
#         # 6.2.1 
#         # α value is determined using a pseudorandom number
#         # it is based on the difference minimization 
#         # source: Albert's tesis
        
#         # 6.2.2.1 get a alfa_n randomly 
#         # eq 1-29
#         list_theta=abs(np.random.rand(1)-Palf_norm)
#         theta_min=min(list_theta)
#         theta_n=np.array(list_theta).argmin()   
        
#         # 6.2.2.2 create beta
#         # for n2 in range(1,self.nbeta):
#             # beta = (n2-1)*2*math.pi/(self.nbeta-1)

#         # self.nphi = 360
                
#         phi = [(n2)*2*math.pi/(n_phi) for n2 in range(1,n_phi)]
#         phi = np.array(phi)

#         # 6.2.2.3 calculate the P(theta,phi)
#         # theta is out-angle
#         # where phi is rotating angle
#         # for i in phi:
#         p_phi = np.zeros((n_phi,1))
#         for id, phi_angle in enumerate(phi):
#             p_phi[id] = np.sum(P[:,id])
#             # pp = p_phi[id]
        
#         P_cum=np.cumsum(p_phi)
#         P_cum=P_cum/P_cum[n_phi-1]


#         # 6.2.2.4 get a beta_n randomly 
#         # Choose random alfa and beta teking into account the 
#         # probability distribution P.
        
#         list_phi=abs(np.random.rand(1)-P_cum)
#         phi_min=min(list_phi)
#         phi_n=np.array(list_phi).argmin()  

#         theta=(theta_n-1)*math.pi/(n_theta-1)
#         phi=(phi_n-1)*2*math.pi/(n_phi-1)


#         # 6.2.2 
#         # calculate the I0, Qb, Ub, Vo

#         T1, T2, T3, T4 = self.mie_cylinder_cal_T(phi,an1,an2,bn1,bn2,theta_n)
#         #  TODO 
#         # check if it is true
#         #  just based on mermory

#         M = self.mie_cylinder_Smatrix(T1,T2,T3,T4)
        
#         # rotation for cylinder scattering

#         # l =np.array([1,0,0])
#         vi = np.cross(np.array(l),d_vectors.reshape(1,3))
#         #  TODO: strange dimension here

#         angle_difference = np.arccos(np.dot(d_vectors.ravel(),vi.ravel())/(np.linalg.norm(d_vectors.ravel())*np.linalg.norm(vi.ravel())))
#         # angle_difference =ad[0]
#         R_matrix = np.array([1,0,0,0,0,np.cos(2*angle_difference),np.sin(2*angle_difference),0,0,-np.sin(2*angle_difference),np.cos(2*angle_difference),0,0,0,0,1])
#         R_matrix = R_matrix.reshape(4,4)
#         # R_matrix = np.array([1,0,0,0],
#         #                     [0,np.cos(2*angle_difference),np.sin(2*angle_difference),0],
#         #                     [0,-np.sin(2*angle_difference),np.cos(2*angle_difference),0],
#         #                     [0,0,0,1]
#         #                     )

#         Sout = M@R_matrix@self.stokes
        
#         # stokes=np.array([I0, Qb, Ub, Vo])

#         return theta, phi, Sout
    
#     def mie_cylinder_miut(self,d,ca,zeta,nzeta,n_theta,k,an1,an2,bn1,bn2,stokes_in,qsca2):
#         # d : diameter
#         # CA : density
#         c_compute = mymath()
#         numerator = 0
#         denominator = 0
#         # new_q_sca = np.zeros((nzeta,1))

#         a = np.ceil(zeta/(np.pi/nzeta))
#         # print(a)
#         for j in range(1,n_theta):
#             theta = j*2*np.pi/n_theta
#             T1,T2,T3,T4 = self.mie_cylinder_cal_T(theta,an1,an2,bn1,bn2,a)
#             # print(T1,T2,T3,T4)
#             M=2/(np.pi*k*d/2*np.sin(zeta))*self.mie_cylinder_Smatrix(T1,T2,T3,T4)
#             V1 = M @ stokes_in.reshape(4,1)
#             numerator = numerator + V1[0]
#             V2 = M @ np.array([[1],[1],[0],[0]])
#             denominator = denominator + V2[0]
#             kk = c_compute.complex_div(numerator,denominator)
#             # print(kk)
#             # print(qsca2)
#             # print(ca)
#             # new_q_sca[i] = d*ca*qsca2*numerator/denominator*10000
#             q_sca = d*ca*np.real(qsca2)*np.real(kk)
#             # 10000 is the factor for r, also for d
#             # new_q_sca[i] = c_compute.complex_multi(qsca2,kk)
        
#             # print("miu_t = {}".format(q_sca))
#         return q_sca


#         # q_sca = np.zeros((n_zeta))
#         # for i in range(1,n_zeta):
#         #     zeta = i*np.pi/n_zeta
#         #     numerator = 0
#         #     denominator = 0
#         #     for j in range(1,n_theta):
#         #         theta = i*2*np.pi/n_theta
#         #         T1,T2,T3,T4 = self.mie_cylinder_cal_T(theta,an1,an2,bn1,bn2,i)
#         #         # print(T1,T2,T3,T4)
#         #         M=2/(np.pi*k*d/2*np.sin(zeta))*self.mie_cylinder_Smatrix(T1,T2,T3,T4)
#         #         V1 = M @ stokes_in.reshape(4,1)
#         #         numerator = numerator + V1[0]
#         #         V2 = M @ np.array([[1],[1],[0],[0]])
#         #         denominator = denominator + V2[0]
#         #     kk = c_compute.complex_div(numerator,denominator)
#         #     # print(kk)
#         #     # print(d)
#         #     # print(ca)
#         #     q_sca[i] = d*ca*np.real(qsca2)*np.real(kk)
#         #     # new_q_sca[i] = c_compute.complex_multi(qsca2,kk)
        
#         #     # print(d)
#         # return q_sca[0:n_zeta-1]
     
#     def reflection_interfaces():
#         pass

class position:
    # Based on Meridian plane geometry theory

    def __init__(self,p_vectors,d_vectors):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.cuda.FloatTensor
            # print("using GPU:",torch.cuda.get_device_name(0))
        else:
            self.device =torch.device('cpu')
            self.dtype = torch.FloatTensor
            # print("Using CPU")

        if torch.is_tensor(p_vectors):
        # p_vectors: xi,yi,zi
        # d_vectors: ux,uy,uz
            self.p_vectors=p_vectors
            self.d_vectors=d_vectors
            self.xi = self.p_vectors[:,0]
            self.yi = self.p_vectors[:,1]
            self.zi = self.p_vectors[:,2]
            self.ux = self.d_vectors[:,0]
            self.uy = self.d_vectors[:,1]
            self.uz = self.d_vectors[:,2]
        else:

            self.p_vectors=torch.tensor(p_vectors).type(self.dtype)
            self.d_vectors=torch.tensor(d_vectors).type(self.dtype)
            self.xi = self.p_vectors[:,0]
            self.yi = self.p_vectors[:,1]
            self.zi = self.p_vectors[:,2]
            self.ux = self.d_vectors[:,0]
            self.uy = self.d_vectors[:,1]
            self.uz = self.d_vectors[:,2]

    def change_complete(self,alfa,beta):
        # sub_z=torch.sqrt(1-self.uz**2)
        # sphere_scattering
        # calculate the out-position
        if self.ux.ndim<=1:
            self.ux = self.ux.unsqueeze(1)
            self.uy = self.uy.unsqueeze(1)
            self.uz = self.uz.unsqueeze(1)
            
        elif self.ux.ndim==2:
            pass
        sub_z=torch.sqrt(1-self.uz**2)
        uxo=self.ux*torch.cos(alfa)-torch.sin(alfa)*(self.ux*self.uz*torch.cos(beta)-self.uy*torch.sin(beta))/sub_z
        uyo=self.uy*torch.cos(alfa)-torch.sin(alfa)*(self.uy*self.uz*torch.cos(beta)+self.ux*torch.sin(beta))/sub_z
        uzo=self.uz*torch.cos(alfa)+torch.sin(alfa)*torch.cos(beta)*sub_z

        return torch.cat((uxo,uyo,uzo),dim=1) 
    
    def change_simple(self,alfa,beta):
        # sphere_scattering
        # used when u_z0 is near to 1
        if self.ux.ndim<=1:
            self.ux = self.ux.unsqueeze(1)
            self.uy = self.uy.unsqueeze(1)
            self.uz = self.uz.unsqueeze(1)
        elif self.ux.ndim==2:
            pass

        uxo=torch.sin(alfa)*torch.cos(beta)
        uyo=torch.copysign(torch.sin(alfa)*torch.sin(beta),self.uz)
        # math.sin(alfa)*math.sin(beta)*np.sign(self.zi)
        uzo=torch.copysign(torch.cos(alfa),self.uz)
        
        return torch.cat((uxo,uyo,uzo),dim=1) 
        # torch.tensor([uxo,uyo,uzo])
    
    def random_walk(self,miu_t):
        if miu_t<0.01:
            delta_s = torch.tensor(0.01).type(self.dtype)
            
        else:
            delta_s=-torch.log(torch.rand(1).type(self.dtype))/miu_t

        
        x_dash=self.xi+self.ux*delta_s
        y_dash=self.yi+self.uy*delta_s
        z_dash=self.zi+self.uz*delta_s
        return torch.tensor([x_dash,y_dash,z_dash]).type(self.dtype)

    def random_walks(self, miu_t):
        delta_s=-torch.log(torch.rand(1).type(self.dtype))/miu_t
        p = self.p_vectors
        p[:,0]=p[:,0]+self.ux*delta_s
        p[:,1]=p[:,1]+self.uy*delta_s
        p[:,2]=p[:,2]+self.uz*delta_s
        return p
    
class transmission(photons_event, position):

    def __init__(self,miu_s,miu_a,d,th,x_max,y_max,number_photon,Scattering_matrix,stokes_in,Palf_norm,nalfa,nbeta,p_vectors,d_vectors,pattern,beam_radius,batch_size):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.cuda.FloatTensor
            # print("using GPU:",torch.cuda.get_device_name(0))
        else:
            self.device =torch.device('cpu')
            self.dtype = torch.FloatTensor

        self.miu_s = miu_s
        self.miu_a = miu_a
        self.miu_t = miu_a+miu_s
        self.d = torch.tensor(d).type(self.dtype)
        self.th = th
        # depth = D/ ut
        self.x_max=torch.tensor(x_max).type(self.dtype)
        self.y_max=torch.tensor(y_max).type(self.dtype)
        self.number_photon=number_photon
        self.pattern=pattern
        self.beam_radius=beam_radius
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.cuda.FloatTensor
            # print("using GPU:",torch.cuda.get_device_name(0))
        else:
            self.device =torch.device('cpu')
            self.dtype = torch.FloatTensor
            # print("Using CPU")
        # I have to use the function in photon event, sorry ~
        # super class 
        
        # super().__init__(Scattering_matrix,stokes_in,Palf_norm,nalfa,nbeta,p_vectors,d_vectors)
        # photon_event.__init__(self,Scattering_matrix,stokes_in,Palf_norm,nalfa,nbeta)
        photons_event.__init__(self,Scattering_matrix,stokes_in,Palf_norm,nalfa,nbeta,batch_size)
        position.__init__(self,p_vectors,d_vectors)
        # super(position, self).__init__(p_vectors,d_vectors)
        # super(photon_event, self).__init__(Scattering_matrix,stokes_in,Palf_norm,nalfa,nbeta)
    
    def camera_initialization(self,resolution_x,resolution_y):
        camera_vectors = torch.zeros(resolution_x,resolution_y,4).type(self.dtype)
        # camera_vectors_complex = camera_vectors.type(torch.complex64)
        return camera_vectors 
   
    def meida_initialization(self):
        
        self.t_vectors = torch.zeros(4,1).type(self.dtype)
        self.b_vectors = torch.zeros(4,1).type(self.dtype)
        
        # # 5.1.1 parameters for transmission
        # self.t_vectors = t_vectors.type(torch.complex64)
        # self.t_vectors.to(self.device)
        # # 5.1.2. parameters for backscattering
        # self.b_vectors = b_vectors.type(torch.complex64)
        # self.b_vectors.to(self.device)
        # 5.1.3 light path of thickness
        # self.depth_z = self.d/self.miu_t

    def measurement_initialization(self,bacth_size):
        d_measure = torch.zeros(3,1,bacth_size).type(self.dtype)
        count = torch.zeros(4,1,bacth_size).type(self.dtype)
        return d_measure ,count
    
    def photon_initialization(self):   
        # position
        p_vectors = np.zeros((3,1))
        if self.pattern == 'ideal':
            # move to the centre
            p_vectors [0] = p_vectors [0]+ self.x_max/2
            p_vectors [1] = p_vectors [1]+ self.y_max/2
            p_vectors [2] = p_vectors [2]+ 0
        elif self.pattern == 'evenly':
            rand_x= 2*self.beam_radius*np.random.rand(1)-self.beam_radius
            # [0,1) --> [-beam_radius,beam_radius)
            p_vectors [0] = p_vectors [0]+ self.x_max/2 + rand_x
            y_range = np.sqrt(self.beam_radius**2-rand_x**2)
            rand_y= 2*y_range*np.random.rand(1)-y_range
            p_vectors [1] = p_vectors [1]+ self.y_max/2+ rand_y
            # p_vectors [2] = p_vectors [2]+ 0
            # x,y,z -- 0,1,2
        elif self.pattern == 'GS':
            pass
        else:
            raise Exception("Unspported laser pattern!")
        #
        #  direction
        d_vectors = np.zeros((3,1))
        d_vectors[2] = 1

        # initialized absorption
        albedo = 1
        
        return p_vectors, d_vectors,albedo

    def photons_initialization(self,N):   
        
        P = torch.zeros(N,3).type(self.dtype)
        D = torch.zeros(N,3).type(self.dtype)
        
        if self.pattern == 'ideal':
            # move to the centre
            P[:,0] = P[:,0]+ self.x_max/2
            P[:,1] = P[:,1]+ self.y_max/2
            P[:,2] = P[:,2]+ 0
        
        elif self.pattern == 'evenly':
            for i in range(0,N):
                rand_x= 2*self.beam_radius*np.random.rand(1)-self.beam_radius
                # [0,1) --> [-beam_radius,beam_radius)
                P[i,0] = P[i,0]+ self.x_max/2 + rand_x
                y_range = np.sqrt(self.beam_radius**2-rand_x**2)
                rand_y= 2*y_range*np.random.rand(1)-y_range
                P[i,1] = P[i,1]+ self.y_max/2+ rand_y
        
        elif self.pattern.endswith('gaussian'):
            judge = mymath()
            if judge.is_square(N):
                # m_max = 1
                # n_max = 1 
                # generate the coordinate
                x = np.linspace(-self.beam_radius,self.beam_radius,int(np.sqrt(N)))
                y = np.linspace(-self.beam_radius,self.beam_radius,int(np.sqrt(N)))
                X,Y= np.meshgrid(x,y)

                if self.pattern.startswith('hermite'):
                    m = 4
                    n = 0
                    W = 1
                    radial_part =np.exp(-(X**2 + Y**2)/(W**2))
                    hermite_m = eval_hermite(m,np.sqrt(2)*X/W)
                    hermite_n = eval_hermite(n,np.sqrt(2)*Y/W)

                    intensity = (1/np.sqrt(2**n*np.math.factorial(n)*np.sqrt(np.pi)*W)) *\
                                hermite_m*radial_part*hermite_n
                    intensity = np.abs(intensity)
                    
                elif self.pattern.startswith('laguerre'):
                    # pl mode
                    # https://es.mathworks.com/matlabcentral/fileexchange/72741-laguerre-gauss-visualization
                    p=0
                    l=2
                    # FWHM
                    W=0.2
                    # Term 1
                    t = (X**2 + Y**2)/(W**2)
                    Phi = l*np.arctan(Y/X)
                    Term1 =np.power((np.sqrt(2)*np.sqrt(X**2 + Y**2)/W),l)
                    # generalized Laguerre polynomial
                    Term2 =genlaguerre(p,l)(2*(X**2 + Y**2)/(W**2))
                    Term3 = np.exp(-(X**2 + Y**2)/(W**2))
                    Term4 = np.exp(1j*Phi)
                    intensity = np.abs(Term1*Term2*Term3*Term4)
                    
                elif self.pattern.startswith('ince'):
                    a = 0.5
                    # maximum radial mode index
                    m = 2
                    # maximun azimuthal mode index 
                    n = 2
                    # N - factor
                    normalization = np.sqrt((2**a*np.math.factorial(m+n))/(np.math.factorial(n)*np.math.factorial(m)))
                    
                    intensity = (
                        np.sqrt(2/np.pi)*normalization*
                        X**m * Y**n *
                        np.exp(-0.5*(X**2 + Y**2))*
                        genlaguerre(n,m)(X**2 + Y**2)*
                        hyp2f1(-a,n+m+1,m+1,-(X**2 + Y**2))
                    )
                    intensity = np.abs(intensity)

                else:
                    sigma_sqaure= 0.02 
                    # caculate the instensity
                    intensity=np.exp(-(X**2+Y**2)/(2*sigma_sqaure))
                    # Normalize internsity to get probability distribution
                prob_distribution = intensity/np.sum(intensity)
                indices = np.random.choice(np.arange(len(X.flatten())),size=N,p=prob_distribution.flatten())
                x_photons, y_photons =np.unravel_index(indices,X.shape)
                P[:,0] = X[x_photons, y_photons]
                P[:,1] = Y[x_photons, y_photons]

                # move to the centre
                P[:,0] = P[:,0]+ self.x_max/2
                P[:,1] = P[:,1]+ self.y_max/2
                # return X[x_photons, y_photons], Y[x_photons, y_photons]
            else:
                raise Exception("Unspported photons numbers under this mode!")           
        
        elif self.pattern.endswith('structured'):
            x = np.linspace(-self.beam_radius,self.beam_radius,int(np.sqrt(N)))
            y = np.linspace(-self.beam_radius,self.beam_radius,int(np.sqrt(N)))
            X,Y= np.meshgrid(x,y)

            if self.pattern.startswith('grid'):
                if self.pattern.startswith('gridx'):
                    is_x =True
                    is_y =False
                elif self.pattern.startswith('gridy'):
                    is_x =False
                    is_y =True
                elif self.pattern.startswith('gridinclined'):
                    is_x =False
                    is_y =False
                grid_g = mymath()
                # grid spacing
                dx= 2
                dy= 5
                kx = 30
                ky = 1
                # average intensity
                I0 = 5
                # Amplitude
                A =1
                # Intensity
                if is_x:
                    intensity_pattern = grid_g.tozero(np.mod(X,dx))
                elif is_y:
                    intensity_pattern = grid_g.tozero(np.mod(Y,dy))
                else:
                    # Set the angle of the diagonal stripes (in radians)
                    angle = np.radians(60)
                    # Compute the diagonal distance from the origin
                    diagonal_distance = X * np.cos(angle) + Y * np.sin(angle)
                    # Set the stripe spacing
                    stripe_spacing = 5
                    # Generate the diagonal stripes
                    intensity_pattern = np.sin(2 * np.pi * diagonal_distance / stripe_spacing)
                    # intensity_pattern = grid_g.tozero(np.mod(X,dx)) + grid_g.tozero(np.mod(Y,dy))

                intensity = np.abs(intensity_pattern)

            elif self.pattern.startswith('sin'):
                # frequency information --  wave number
                kx = 2*1e9
                ky = 3
                phi = 0
                I0 = 5
                A =1
                #  pattern 
                intensity_pattern = I0 + A* np.sin(kx*X + 2*kx*X  + phi)
                # intensity_pattern = I0 + A* np.sin(kx*X + 2*kx*X + 3*kx*X + 4*kx*X + phi)
                # intensity_pattern = I0 + A* np.sin(kx*X + 2*kx*X +  3*kx*X + 4*kx*X +  5*kx*X + 6*kx*X + phi)
                intensity = np.abs(intensity_pattern)

            elif self.pattern.startswith('Lcosrandom'):
                intensity = np.random.rand(32,32)
            else:
                raise Exception("Unspported photons numbers under this mode!")  

            prob_distribution = intensity/np.sum(intensity)
            indices = np.random.choice(np.arange(len(X.flatten())),size=N,p=prob_distribution.flatten())
            x_photons, y_photons =np.unravel_index(indices,X.shape)
            P[:,0] = X[x_photons, y_photons]
            P[:,1] = Y[x_photons, y_photons]

            # move to the centre
            P[:,0] = P[:,0]+ self.x_max/2
            P[:,1] = P[:,1]+ self.y_max/2
            # return X[x_photons, y_photons], Y[x_photons, y_photons]
            
        D[:,2] = 1
        albedo = torch.ones(N,1).type(self.dtype)
        
        return P, D, albedo
    
    def mask_det_position(self,p_vectors,flag):
        # 5.3 determinate whether is inside of the "tank"
        # or not
        # TODO: set names for differnt position?
        len = p_vectors.size()[0]
        # mask = torch.zeros(len,1).type(self.dtype)
        for i in torch.arange(0,len):
            flag[i]=self.det_position2(p_vectors[i,:],flag[i])

        return flag

    def albedo_condition(self, albedo, flag):
        return (flag==1) & (albedo<=self.th)

    def mask_det_albedo(self,albedo,flag):
        # 5.4 determinate whether the instensity is enough
        # for following propagation or not
        # flag=self.mask_det_position2(p_vectors)
        
        mask=self.albedo_condition(albedo,flag)
        flag[mask]=0

        return flag

    def det_position2(self,p_vectors,flag):
        # 5.3 determinate whether is inside of the "tank"
        # or not
        # TODO: set names for differnt position?
        if p_vectors.ndim==1:
            pass
        else: 
            p_vectors = p_vectors.squeeze(dim=0)
        if flag==1:
            # since the size of p_vectors is [1,3]
            if (p_vectors[2]>0)&(p_vectors[2]<self.d):
                # z >0 and z< zmax
                if (p_vectors[1]>0)&(p_vectors[1]<self.y_max):
                # y >0 and y< ymax 
                    if (p_vectors[0]>0)&(p_vectors[0]<self.x_max):
                    # y >0 and y< ymax 
                        return 1
                    else: 
                        return 0
                else: 
                    return 0
            else:
                return 0
        else:
            return 0
        
    def det_position(self,p_vectors):
        # 5.3 determinate whether is inside of the "tank"
        # or not
        # TODO: set names for differnt position?
        if (p_vectors[2]>0)&(p_vectors[2]<self.d):
            # z >0 and z< zmax
            if (p_vectors[1]>0)&(p_vectors[1]<self.y_max):
               # y >0 and y< ymax 
                if (p_vectors[0]>0)&(p_vectors[0]<self.x_max):
                # y >0 and y< ymax 
                    return True
                else: 
                    return False
            else: 
                return False
        else:
            return False

    def det_albedo(self,p_vectors,albedo):
        # 5.4 determinate whether the instensity is enough
        # for following propagation or not
        flag=self.det_position(p_vectors)
        if flag:
            if albedo>=self.th:
                return True
            else:
                return False
        else:
            return False
    
    def out_direction(self,alfa,beta,p_vectors,d_vectors):
        direction = position(p_vectors,d_vectors)
        
        # uz --> d_vectors[2]
        if (1-abs(d_vectors[2])<=1e-12):
            d_vectors = direction.change_simple(alfa,beta)
        else:
            d_vectors = direction.change_complete(alfa,beta)
            # ux_a,uy_a,uz_a = position(self)   
        
        return d_vectors
    
    # def update_polarized_states(self,alfa,beta,ux,uy,uz,I0,Q0,U0,V0):
    def update_polarized_states(self,alfa,beta,p_vectors,d_vectors,stokes):
        
        
        ux = d_vectors[:,0]
        uy = d_vectors[:,1]
        uz = d_vectors[:,2]

        I0 = stokes[:,0]
        Q0 = stokes[:,1]
        U0 = stokes[:,2]
        V0 = stokes[:,3]
        
        if ux.ndim<=1:
            ux = ux.unsqueeze(dim=1)
            uy = uy.unsqueeze(dim=1)
            uz = uz.unsqueeze(dim=1)
        else: 
            pass

        if I0.ndim<=1:
            I0 = I0.unsqueeze(dim=1)
            Q0 = Q0.unsqueeze(dim=1)
            U0 = U0.unsqueeze(dim=1)
            V0 = V0.unsqueeze(dim=1)
        else: 
            pass

        # 5.5 update the polarized states
        # 5.5.1 calculate the rotational matrix 
        # Input:
        #       alfa,beta: out angle 
        #       d_vectors ([ux,uy,uz]): direction after scattering
        #       stokes ([I0,Q0,U0,V0]): initialized Polarized states
        # output:
        #       Io,Qo,Uo,Vo: Output Polarized states
        # source: 
        # "Three Monte Carlo programs of polarized light transport into scattering media: part I"
        #  Jessica C. Ramella-Roman, Scott A. Prahl, Steve L. Jacques 
        # page 12, eq 15

        # !!!!
        # But we have to discuss the value of denomonator of Eq 15
        # ideas from Albert
        #z = zi-np.log(np.random)/self.miu_t
        
        # 5.5.1.0 initialized the direction
        
        # if 1-abs(uz)<=1e-12:
        #     initialized_direction = methodcaller('change_simple')
        #     ux_a,uy_a,uz_a = position(self)
        # else:
        #     initialized_direction = methodcaller('change_complete')
        #     ux_a,uy_a,uz_a = position(self)    

        get_direction = position(p_vectors, d_vectors)
        d_vectors_new=torch.zeros_like(d_vectors).type(self.dtype)
        Qo=torch.zeros_like(Q0).type(self.dtype)
        Vo=torch.zeros_like(V0).type(self.dtype)

        # situation 1
        mask1 = 1-abs(uz)<=1e-12
        cmask1 = ~mask1
        dv1= get_direction.change_simple(alfa,beta)
        dv2 = get_direction.change_complete(alfa,beta)

        # torch.nan_to_num
        d_vectors_new1 = dv1*mask1
        d_vectors_new2 = dv2*cmask1

        d_vectors_new = torch.nan_to_num(d_vectors_new1) + torch.nan_to_num(d_vectors_new2)
        
        # [batch_size,1]
        ux_a = d_vectors_new[:,0].unsqueeze(dim=1)
        uy_a = d_vectors_new[:,1].unsqueeze(dim=1)
        uz_a = d_vectors_new[:,2].unsqueeze(dim=1)
        D = torch.cat((ux_a,uy_a,uz_a),dim=1)

        nu_cos_gama  = -uz+uz_a*torch.cos(alfa)

        # situation 2
        mask2 = (1-torch.cos(alfa)**2)*(1-uz_a**2)>=0
        dcg1 = torch.sqrt((1-torch.cos(alfa)*torch.cos(alfa))*(1-uz_a**2))
        cmask2 = ~mask2
        dcg2 = torch.sqrt(-(1-torch.cos(alfa)**2)*(1-uz_a**2)) 

        den_cos_gama1 =  mask2*dcg1
        den_cos_gama2 = cmask2*dcg2
        den_cos_gama = torch.nan_to_num(den_cos_gama1) + torch.nan_to_num(den_cos_gama2)

        # situation 3

        mask3 = den_cos_gama == 0
        cmask3 = ~mask3

        mask_31 = alfa == 0
        mask31 =  mask3 & mask_31
        cg1=torch.copysign(torch.cos(beta),torch.sin(beta))
        mask_32 = alfa == torch.pi
        mask32 =  mask3 & mask_32
        cg2=torch.copysign(torch.cos(torch.pi-beta),torch.sin(beta))
        mask_33 = uz_a>0.9999
        mask33 =  mask3 & mask_33
        cg3=torch.copysign(torch.cos(torch.atan(uy/ux)),uy/ux)
        
        
        mask_34 = ~(mask_31|mask_32|mask_33)
        mask34 = mask3 & mask_34
        cg4=-torch.copysign(torch.cos(torch.atan(uy/ux)),uy/ux)

        mask41 = (beta>torch.pi)
        mask42 = (beta<2*torch.pi)
        mask_4 = mask41 & mask42
        mask5 = mask_4 & cmask3
        cg5 = -nu_cos_gama/den_cos_gama
        
        mask6 = ~mask_4 & cmask3
        cg6 = nu_cos_gama/den_cos_gama

        cosgama1 = cg1*mask31 
        cosgama2 = cg2*mask32 
        cosgama3 = cg3*mask33
        cosgama4 = mask34*cg4
        cosgama5 = mask5*cg5 
        cosgama6 = mask6* cg6

        cosgama = torch.nan_to_num(cosgama1) + torch.nan_to_num(cosgama2) + torch.nan_to_num(cosgama3) + torch.nan_to_num(cosgama4) + torch.nan_to_num(cosgama5) + torch.nan_to_num(cosgama6)

        cosgama = torch.clamp(cosgama, min=-1, max=1)

        Qo=(2*cosgama**2-1)*Q0-2*torch.sqrt(1-cosgama**2)*cosgama*U0
        Uo=(2*cosgama**2-1)*U0+2*torch.sqrt(1-cosgama**2)*cosgama*Q0
        
        # Normalize Io,Qo,Uo,Vo
        Qo=Qo/I0
        Uo=Uo/I0
        Vo=V0/I0
        Io=torch.ones_like(Q0).type(self.dtype)
        S=torch.cat((Io,Qo,Uo,Vo),dim=1)

        
        # for i in range(uz.size()[0]):
        #     if 1-abs(uz[i])<=1e-12:
        #         d_vectors_new[i,:] = get_direction.change_simple(alfa[i],beta[i])
        #     else:
        #         d_vectors_new[i,:] = get_direction.change_complete(alfa[i],beta[i])

        #     ux_a=d_vectors_new[i,0]
        #     uy_a=d_vectors_new[i,1]
        #     uz_a=d_vectors_new[i,2]

        #     nu_cos_gama  = -uz+uz_a*torch.cos(alfa[i])

        #     # TODO !!!!!
        
        #     if (1-torch.cos(alfa[i])*torch.cos(alfa[i]))*(1-uz_a**2)>=0:
        #         den_cos_gama = torch.sqrt((1-torch.cos(alfa[i])*torch.cos(alfa[i]))*(1-uz_a**2))
        #     else:
        #         den_cos_gama = torch.sqrt(-(1-torch.cos(alfa[i])*torch.cos(alfa[i]))*(1-uz_a**2)) 
        
        #     # 5.5.1.2 Deatiled discussion !!!
        #     if den_cos_gama == 0:
        #         # 5.5.1.2.1 situation 1: cos(alfa)==1 -- alfa=0
        #         if alfa[i]==0:
        #             cosgama=torch.copysign(torch.cos(beta[i]),torch.sin(beta[i]))
        #             # math.cos(beta)*np.sign(math.sin(beta))
        #             # 5.5.1.2.2 situation 2: cos(alfa)==1 -- alfa=pi
        #         elif alfa[i]==torch.pi:
        #             cosgama=torch.copysign(torch.cos(torch.pi-beta[i]),torch.sin(beta[i]))
        #             # math.cos(math.pi-beta)*np.sign(math.sin(beta))
        #             # 5.5.1.2.3 situation 3: out of boundary
        #             # TODO: where are they from??
        #         elif uz_a>0.9999:
        #             cosgama=torch.copysign(torch.cos(torch.atan(uy[i]/ux[i])),uy[i]/ux[i])
        #             # math.cos(math.atan(uy/ux))*np.sign(uy/ux)
        #         else:
        #             cosgama=-torch.copysign(torch.cos(torch.atan(uy[i]/ux[i])),uy[i]/ux[i])
        #             # math.cos(math.atan(uy/ux))*np.sign(uy/ux)
        #     else:
        #         # 5.5.1.2.4 situation 4: discuss the positivity of numonator
        #         if (beta[i]>torch.pi) & (beta[i]<2*torch.pi):
        #             cosgama = -nu_cos_gama/den_cos_gama
        #         else:
        #             cosgama = nu_cos_gama/den_cos_gama
        #         # 5.5.1.2.5 situation 5:set the value limit for cos(gama)
        #         if cosgama < -1:
        #             cosgama = torch.tensor(-1).type(self.dtype)
        #         elif cosgama > 1:
        #             cosgama = torch.tensor(1).type(self.dtype)

        #     # 5.5.1.3 calculate the Output of the Stokes vector
        #     # TODO  where it is from???
        #     Qo[i]=(2*cosgama**2-1)*Q0[i]-2*torch.sqrt(1-cosgama**2)*cosgama*U0[i]
        #     Uo[i]=(2*cosgama**2-1)*U0[i]+2*torch.sqrt(1-cosgama**2)*cosgama*Q0[i]

        # # 5.5.1.3.1 Normalize Io,Qo,Uo,Vo
        # Qo=Qo/I0
        # Uo=Uo/I0
        # Vo=V0/I0
        # Io=1
        # S=torch.tensor([Io,Qo,Uo,Vo]).type(self.dtype)


        # nan_probe =torch.isnan(S)
        # # print(nan_probe)
        # if torch.sum(nan_probe)==0:
        #     flag_c=True
        #     D=torch.tensor([ux_a,uy_a,uz_a]).type(self.dtype)
        #     return S,D,flag_c
        # else:
        #     print("unstable photon, drop it directly!!")
        #     print(stokes)
        #     flag_c=False

            # return stokes,d_vectors,flag_c

        nan_probe =torch.isnan(S)
        return S,D,nan_probe
    
    def update_albedo(self,albedo,miu_s,miu_a):
        # 5.6 update_albedo
        albedo= albedo*miu_s/(miu_s+miu_a)
        return albedo

    def update_albedo_cylinder(self,albedo,miu_s,miu_a,miu_t):
        # 5.6 update_albedo
        mius = (miu_s[0]+miu_s[1])/2
        miua = (miu_a[0]+miu_a[1])/2
        albedo= albedo*mius/(mius+miua)
        return albedo

    def last_action_photons(self,p_vectors, d_vectors,stokes,albedo,stokes_camera):
        # # 5.7 The last rotation
        # source: 
        # "Three Monte Carlo programs of polarized light transport into scattering media: part I"
        # Jessica C. Ramella-Roman, Scott A. Prahl, Steve L. Jacques 
        # page 15, Photon life and boundaries

        # print(p_vectors)

        x = p_vectors[:,0]
        y = p_vectors[:,1]
        z = p_vectors[:,2]

        ux = d_vectors[:,0]
        uy = d_vectors[:,1]
        uz = d_vectors[:,2]

        I = stokes[:,0]
        Q = stokes[:,1]
        U = stokes[:,2]
        V = stokes[:,3]

        Itb = stokes_camera[:,:,0]
        Qtb = stokes_camera[:,:,1]
        Utb = stokes_camera[:,:,2]
        Vtb = stokes_camera[:,:,3]

        mask1 = z<=0
        # update the x & y
        x=x-ux*z/uz 
        y=y-uy*z/uz
        
        mask21 = y>=0
        mask22 = y<=self.y_max
        mask23 = x>=0
        mask24 = x<=self.x_max
        
        mask2 = mask21 & mask22 & mask23 & mask24

        # if four conditions all meet ...
        final_phi=torch.arctan(uy/(ux+1e-6))
        Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
        Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
        # camera mapping
        mapping = torch.zeros(Itb.shape[0],Itb.shape[1],y.size()[0]).type(self.dtype)
        print("ysize{}".format(y.size()[0]))
        y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
        x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
        xx= Itb.shape[1]-y_coordinate_limit.int()-1
        yy= Itb.shape[0]-x_coordinate_limit.int()-1


        # if p_vectors.grad is None:
        #     return stokes_camera
        # else:
        if y.size()[0]>1:
            if (((xx>=0)&(yy>=0)&(xx<=Itb.shape[0])&(yy<=Itb.shape[0])).sum()>0):
                for i in range(0,y.size()[0]):
                    if y.ndim ==1:
                        mapping[abs(xx),abs(yy),i]=1
                    else:
                        mapping[abs(xx[i]),abs(yy[i]),i]=1

        if y.size()[0]==1:
            if ((xx>=0)&(yy>=0)&(xx<=Itb.shape[0])&(yy<=Itb.shape[0])):
                for i in range(0,y.size()[0]):
                    if y.ndim ==1:
                        mapping[abs(xx),abs(yy),i]=1
                    else:
                        mapping[abs(xx[i]),abs(yy[i]),i]=1
        else:
            pass

        adding_I = (I.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_Q = (Qf.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_U = (Uf.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_V = (V.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping

        adding_I = torch.nan_to_num(adding_I)
        adding_Q = torch.nan_to_num(adding_Q)
        adding_U = torch.nan_to_num(adding_U)
        adding_V = torch.nan_to_num(adding_V)

        # Situation for this calculation
        mask3 = mask1&mask2
        # S_out1 =  S * mask3
        adding_I = adding_I[:,:,mask3]
        adding_Q = adding_Q[:,:,mask3]
        adding_U = adding_U[:,:,mask3]
        adding_V = adding_V[:,:,mask3]

        Itb=Itb+adding_I.sum(dim=2)
        Qtb=Qtb+adding_Q.sum(dim=2)
        Utb=Utb+adding_U.sum(dim=2)
        Vtb=Vtb+adding_V.sum(dim=2)

        S = torch.stack((Itb,Qtb,Utb,Vtb),dim=2)

        return S
        # if (z<=0):
        #     x=x-ux*z/uz 
        #     y=y-uy*z/uz

        #     if (y>=0)&(y<=self.y_max)&(x>=0)&(x<=self.x_max):
        #         # final rotation angle for backscattered photon
        #         # backward direction
        #         final_phi=torch.arctan(uy/(ux+1e-6))
        #         Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
        #         Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
                
        #         # camera mapping
        #         mapping = torch.zeros(Itb.shape[0],Itb.shape[1]).type(self.dtype)
        #         y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
        #         x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
        #         xx= Itb.shape[1]-int(y_coordinate_limit)-1
        #         yy= Itb.shape[0]-int(x_coordinate_limit)-1
        #         # for i in range()
        #         mapping[xx,yy]=1
                
        #         # print(Itb.shape[1]-int(y_coordinate_limit.item())-1)
        #         # print(Itb.shape[0]-int(x_coordinate_limit.item())-1)
        #         # imaging
        #         Itb=Itb+I*albedo*mapping
        #         Qtb=Qtb+Qf*albedo*mapping
        #         Utb=Utb+Uf*albedo*mapping
        #         Vtb=Vtb+V*albedo*mapping
        #         # TODO
        #         # for reflection
        #         # final_phi=-np.arctan(uy/ux)
        #         # maybe we need a mode to control it

        #         S_out = torch.stack([Itb,Qtb,Utb,Vtb])
        # else:
        #     S_out = stokes_camera

        # return S_out
    
    def last_action_photon(self,p_vectors, d_vectors,stokes,albedo,stokes_camera):
        # # 5.7 The last rotation
        # source: 
        # "Three Monte Carlo programs of polarized light transport into scattering media: part I"
        # Jessica C. Ramella-Roman, Scott A. Prahl, Steve L. Jacques 
        # page 15, Photon life and boundaries

        x = p_vectors[0]
        y = p_vectors[1]
        z = p_vectors[2]

        ux = d_vectors[0]
        uy = d_vectors[1]
        uz = d_vectors[2]

        I = stokes[0]
        Q = stokes[1]
        U = stokes[2]
        V = stokes[3]

        Itb = stokes_camera[0,:,:]
        Qtb = stokes_camera[1,:,:]
        Utb = stokes_camera[2,:,:]
        Vtb = stokes_camera[3,:,:]

        if (z<=0):
            x=x-ux*z/uz 
            y=y-uy*z/uz

            if (y>=0)&(y<=self.y_max)&(x>=0)&(x<=self.x_max):
                # final rotation angle for backscattered photon
                # backward direction
                final_phi=torch.arctan(uy/(ux+1e-6))
                Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
                Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
                
                # camera mapping
                mapping = torch.zeros(Itb.shape[0],Itb.shape[1]).type(self.dtype)
                y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
                x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
                xx= Itb.shape[1]-int(y_coordinate_limit)-1
                yy= Itb.shape[0]-int(x_coordinate_limit)-1
                # for i in range()
                mapping[xx,yy]=1
                
                # print(Itb.shape[1]-int(y_coordinate_limit.item())-1)
                # print(Itb.shape[0]-int(x_coordinate_limit.item())-1)
                # imaging
                Itb=Itb+I*albedo*mapping
                Qtb=Qtb+Qf*albedo*mapping
                Utb=Utb+Uf*albedo*mapping
                Vtb=Vtb+V*albedo*mapping
                # TODO
                # for reflection
                # final_phi=-np.arctan(uy/ux)
                # maybe we need a mode to control it

                S_out = torch.stack([Itb,Qtb,Utb,Vtb])
        else:
            S_out = stokes_camera

        return S_out

    def last_action_transmitted_photons(self,p_vectors, d_vectors,stokes,albedo,stokes_camera):
        # # 5.7 The last rotation
        # source: 
        # "Three Monte Carlo programs of polarized light transport into scattering media: part I"
        # Jessica C. Ramella-Roman, Scott A. Prahl, Steve L. Jacques 
        # page 15, Photon life and boundaries

        x = p_vectors[:,0]
        y = p_vectors[:,1]
        z = p_vectors[:,2]

        ux = d_vectors[:,0]
        uy = d_vectors[:,1]
        uz = d_vectors[:,2]

        I = stokes[:,0]
        Q = stokes[:,1]
        U = stokes[:,2]
        V = stokes[:,3]

        Itb = stokes_camera[:,:,0]
        Qtb = stokes_camera[:,:,1]
        Utb = stokes_camera[:,:,2]
        Vtb = stokes_camera[:,:,3]

        mask11 = z<=0
        mask12 = z>self.d
        mask2 = mask11 & mask12
        x=x-ux*z/uz 
        y=y-uy*z/uz
        mask31 = y>=0
        mask32 = y<=self.y_max
        mask33 = x>=0
        mask34 = x<=self.x_max
        
        mask3 = mask31 & mask32 & mask33 & mask34

        final_phi=-torch.arctan(uy/(ux+1e-6))
        Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
        Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
                
        # camera mapping
        mapping = torch.zeros(Itb.shape[0],Itb.shape[1],y.size()[0]).type(self.dtype)
        y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
        x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
        xx= Itb.shape[1]-y_coordinate_limit.int()-1
        yy= Itb.shape[0]-x_coordinate_limit.int()-1
        # print("cx is {}, cy is {}".format(xx,yy)


        if y.size()[0]>1:
            if (((xx>=0)&(yy>=0)&(xx<=Itb.shape[0])&(yy<=Itb.shape[0])).sum()>0):
                for i in range(0,y.size()[0]):
                    if y.ndim ==1:
                        mapping[abs(xx),abs(yy),i]=1
                    else:
                        mapping[abs(xx[i]),abs(yy[i]),i]=1

        if y.size()[0]==1:
            if ((xx>=0)&(yy>=0)&(xx<=Itb.shape[0])&(yy<=Itb.shape[0])):
                for i in range(0,y.size()[0]):
                    if y.ndim ==1:
                        mapping[abs(xx),abs(yy),i]=1
                    else:
                        mapping[abs(xx[i]),abs(yy[i]),i]=1
        else:
            pass

        adding_I = (I.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_Q = (Qf.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_U = (Uf.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping
        adding_V = (V.reshape(-1,1)*albedo).reshape(1,1,-1)*mapping

        adding_I = torch.nan_to_num(adding_I)
        adding_Q = torch.nan_to_num(adding_Q)
        adding_U = torch.nan_to_num(adding_U)
        adding_V = torch.nan_to_num(adding_V)

        # Situation for this calculation
        mask4 = mask2 & mask3
        adding_I = adding_I[:,:,mask4]
        adding_Q = adding_Q[:,:,mask4]
        adding_U = adding_U[:,:,mask4]
        adding_V = adding_V[:,:,mask4]
                
        Itb=Itb+adding_I.sum(dim=2)
        Qtb=Qtb+adding_Q.sum(dim=2)
        Utb=Utb+adding_U.sum(dim=2)
        Vtb=Vtb+adding_V.sum(dim=2)

        S = torch.stack((Itb,Qtb,Utb,Vtb),dim=2)


        # mask4 = mask2 & mask3
        # S_out1 =  S * mask4
        # cmask4 = ~mask2
        # S_out2 =  stokes * cmask4

        # S_out =  torch.nan_to_num(S_out1) +  torch.nan_to_num (S_out2)

        return S
    
        # if (z<=0):

        if (z<=0)| (z>self.d):
            x=x-ux*z/uz 
            y=y-uy*z/uz

            if (y>=0)&(y<=self.y_max)&(x>=0)&(x<=self.x_max):
                # final rotation angle for backscattered photon
                # backward direction

                final_phi=-torch.arctan(uy/(ux+1e-6))
                Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
                Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
                
                # camera mapping
                mapping = torch.zeros(Itb.shape[0],Itb.shape[1]).type(self.dtype)
                y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
                x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
                xx= Itb.shape[1]-int(y_coordinate_limit)-1
                yy= Itb.shape[0]-int(x_coordinate_limit)-1
                # print("cx is {}, cy is {}".format(xx,yy))
                # for i in range()
                mapping[xx,yy]=1
                
                # print(Itb.shape[1]-int(y_coordinate_limit.item())-1)
                # print(Itb.shape[0]-int(x_coordinate_limit.item())-1)
                # imaging
                Itb=Itb+I*albedo*mapping
                Qtb=Qtb+Qf*albedo*mapping
                Utb=Utb+Uf*albedo*mapping
                Vtb=Vtb+V*albedo*mapping
                # TODO
                # for reflection
                # final_phi=-np.arctan(uy/ux)
                # maybe we need a mode to control it
            S= torch.stack([Itb,Qtb,Utb,Vtb])
            if np.isnan(S).any():
                S=stokes_camera
                # np.array([np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0]))])
        else:
            S=stokes_camera
            # np.array([np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0]))])
        return S


    def last_action_transmitted_photon(self,p_vectors, d_vectors,stokes,albedo,stokes_camera):
        # # 5.7 The last rotation
        # source: 
        # "Three Monte Carlo programs of polarized light transport into scattering media: part I"
        # Jessica C. Ramella-Roman, Scott A. Prahl, Steve L. Jacques 
        # page 15, Photon life and boundaries

        x = p_vectors[0]
        y = p_vectors[1]
        z = p_vectors[2]

        ux = d_vectors[0]
        uy = d_vectors[1]
        uz = d_vectors[2]

        I = stokes[0]
        Q = stokes[1]
        U = stokes[2]
        V = stokes[3]

        Itb = stokes_camera[0,:,:]
        Qtb = stokes_camera[1,:,:]
        Utb = stokes_camera[2,:,:]
        Vtb = stokes_camera[3,:,:]

        if (z<=0)| (z>self.d):
            x=x-ux*z/uz 
            y=y-uy*z/uz

            if (y>=0)&(y<=self.y_max)&(x>=0)&(x<=self.x_max):
                # final rotation angle for backscattered photon
                # backward direction

                final_phi=-torch.arctan(uy/(ux+1e-6))
                Qf=torch.cos(2*final_phi)*Q+torch.sin(2*final_phi)*U
                Uf=torch.cos(2*final_phi)*U-torch.sin(2*final_phi)*Q
                
                # camera mapping
                mapping = torch.zeros(Itb.shape[0],Itb.shape[1]).type(self.dtype)
                y_coordinate_limit=torch.round(y*(Itb.shape[1]-1)/self.y_max)
                x_coordinate_limit=torch.round(x*(Itb.shape[0]-1)/self.x_max)
                xx= Itb.shape[1]-int(y_coordinate_limit)-1
                yy= Itb.shape[0]-int(x_coordinate_limit)-1
                # print("cx is {}, cy is {}".format(xx,yy))
                # for i in range()
                mapping[xx,yy]=1
                
                # print(Itb.shape[1]-int(y_coordinate_limit.item())-1)
                # print(Itb.shape[0]-int(x_coordinate_limit.item())-1)
                # imaging
                Itb=Itb+I*albedo*mapping
                Qtb=Qtb+Qf*albedo*mapping
                Utb=Utb+Uf*albedo*mapping
                Vtb=Vtb+V*albedo*mapping
                # TODO
                # for reflection
                # final_phi=-np.arctan(uy/ux)
                # maybe we need a mode to control it
            S= torch.stack([Itb,Qtb,Utb,Vtb])
            if np.isnan(S).any():
                S=stokes_camera
                # np.array([np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0]))])
        else:
            S=stokes_camera
            # np.array([np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0])),np.zeros((Itb.shape[-1],Itb.shape[0]))])
        return S
    
    def albedo_det_death(self,photons_states,p_vectors, d_vectors, albedo, stokes):
        photons_states = self.mask_det_albedo(albedo,photons_states)
        alive_mask = (photons_states==1)
        # if photons_states.min()!=0:
        #     # mask = torch.ones_like(photons_states,dtype=torch.bool).type(self.dtype)
        #     pass
        # else:
        zero_mask = (photons_states==0)
        # alive_batchsize = alive_mask.sum()
            # zeros_indices = zero_mask.nonzero(as_tuple=True)
            # # mask = torch.ones_like(photons_states,dtype=torch.bool)
            # mask = torch.ones(photons_states.size()[0],dtype=torch.bool).type(self.dtype)
            # mask[zeros_indices[0]]=False
        photons_states = photons_states[alive_mask].reshape(-1,1)

            # mask1 = torch.ones(p_vectors.size()[0],dtype=torch.bool)
            # mask1[zeros_indices]=False

        two_D_mask = mymath()
        p_vectors,d_vectors,albedo,stokes = two_D_mask.torch_two_dimensional_mask(p_vectors,d_vectors,albedo,stokes,alive_mask)

        # p_vectors_x = p_vectors[:,0].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # p_vectors_y = p_vectors[:,1].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # p_vectors_z = p_vectors[:,2].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)

        # p_vectors = torch.cat((p_vectors_x,p_vectors_y,p_vectors_z),dim=1)


        # d_vectors_x = d_vectors[:,0].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # d_vectors_y = d_vectors[:,1].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # d_vectors_z = d_vectors[:,2].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)

        # d_vectors = torch.cat((d_vectors_x,d_vectors_y,d_vectors_z),dim=1)
        
        # albedo = albedo[alive_mask].reshape(self.batch_size,1)

        # s0 = stokes[:,0].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # s1 = stokes[:,1].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # s2 = stokes[:,2].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
        # s3 = stokes[:,3].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)

        # stokes = torch.cat((s0,s1,s2,s3),dim=1)

        return photons_states,p_vectors,d_vectors,albedo,stokes,alive_mask,zero_mask

    def position_det_death(self,photons_states,p_vectors, d_vectors, albedo, stokes, t_camera, b_camera):
        photons_states =self.mask_det_position(p_vectors,photons_states)

        # if photons_states.min()!=0:
        #     mask = torch.ones_like(photons_states,dtype=torch.bool).type(self.dtype)
        #     t_stokes = torch.zeros_like(stokes).type(self.dtype)
        #     b_stokes = torch.zeros_like(stokes).type(self.dtype)
        # else:

        try:        
            zero_mask = (photons_states==0)
            alive_mask = (photons_states==1)
            # photons_states = photons_states[zero_mask]

            # situation 1 
            # transmission
            mask_t = p_vectors[:,2]>self.d
            maskt = mask_t.reshape(zero_mask.size()[0],1) & zero_mask
            two_D_mask = mymath()
            if maskt.sum()==0:
                pass
            else:
                # alive_batchsize = maskt.sum()
                p_vectors_t,d_vectors_t,albedo_t,stokes_t = two_D_mask.torch_two_dimensional_mask(p_vectors,d_vectors,albedo,stokes,maskt)
                # print(p_vectors_t)
                try: 
                    t_camera = self.last_action_transmitted_photons(p_vectors_t, d_vectors_t, stokes_t, albedo_t, t_camera)
                except(RuntimeError):
                    pass
                

            # situation 2 
            # backscattering
            mask_b = p_vectors[:,2]<0
            maskb =  mask_b.reshape(zero_mask.size()[0],1) & zero_mask
            if maskb.sum()==0:
                pass
            else:
                # alive_batchsize = maskb.sum()
            # b_stokes = stokes[mask_b]
                # print(p_vectors_b)
                p_vectors_b,d_vectors_b,albedo_b,stokes_b= two_D_mask.torch_two_dimensional_mask(p_vectors,d_vectors,albedo,stokes,maskb)
                # print(p_vectors_b)
                try: 
                    b_camera = self.last_action_photons(p_vectors_b, d_vectors_b, stokes_b, albedo_b, b_camera)
                except(RuntimeError):
                    pass
            # situation 3
            # photon run off from the side of the tank
            mask_o = zero_mask & (~(maskt | maskb))


            # Save the alive photons
            alive_mask = (photons_states==1)
            photons_states = photons_states[alive_mask].reshape(-1,1)

            p_vectors,d_vectors,albedo,stokes= two_D_mask.torch_two_dimensional_mask(p_vectors,d_vectors,albedo,stokes,alive_mask)


            # p_vectors_x = p_vectors[:,0].unsqueeze(1)[alive_mask].reshape(-1,1)
            # p_vectors_y = p_vectors[:,1].unsqueeze(1)[alive_mask].reshape(-1,1)
            # p_vectors_z = p_vectors[:,2].unsqueeze(1)[alive_mask].reshape(-1,1)

            # p_vectors = torch.cat((p_vectors_x,p_vectors_y,p_vectors_z),dim=1)

            # # print(p_vectors_x.size())
            # # print(p_vectors.size())

            # d_vectors_x = d_vectors[:,0].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
            # d_vectors_y = d_vectors[:,1].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
            # d_vectors_z = d_vectors[:,2].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)

            # d_vectors = torch.cat((d_vectors_x,d_vectors_y,d_vectors_z),dim=1)
            # # d_vectors = d_vectors[alive_mask.repeat(1,3)]
            
            # albedo = albedo[alive_mask].reshape(self.batch_size,1)

            # s0 = stokes[:,0].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
            # s1 = stokes[:,1].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
            # s2 = stokes[:,2].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)
            # s3 = stokes[:,3].unsqueeze(1)[alive_mask].reshape(self.batch_size,1)

            # stokes = torch.cat((s0,s1,s2,s3),dim=1)
                



                # zeros_indices = zero_mask.nonzero(as_tuple=True)
                # # mask = torch.ones_like(photons_states,dtype=torch.bool)
                # mask = torch.ones(photons_states.size()[0],dtype=torch.bool).type(self.dtype)
                # mask[zeros_indices[0]]=False
                # photons_states = photons_states[mask.bool()]

                # # mask1 = torch.ones(p_vectors.size()[0],dtype=torch.bool)
                # # mask1[zeros_indices]=False





                # for indice in zeros_indices[0]:
                #     if p_vectors[indice,2]>self.d:
                #         # transmission
                #         print("transmission photon")
                #         print("stokes vector: ",stokes[indice,:])
                #     elif p_vectors[indice,2]<0:
                #         # backscattering
                #         print("backscattering photon")
                #         print("stokes vector: ",stokes[indice,:])
                #     else:
                #         print("photon run off from the side of the tank")
                
                # p_vectors = p_vectors[mask.bool()]
                # d_vectors = d_vectors[mask.bool()]
                # albedo = albedo[mask.bool()]
                # stokes= stokes[mask.bool()]
        except(RuntimeError):
            alive_mask=torch.ones_like(photons_states).type(self.dtype)
            zero_mask = ~alive_mask
        return photons_states,p_vectors,d_vectors,albedo,stokes,alive_mask,zero_mask,t_camera,b_camera
    
    def Trans(self,resolution_x,resolution_y):
        # 5.7 the whole procedure
        # 5.7.0 initialize the camera
        b_camera = self.camera_initialization(resolution_x,resolution_y)
        t_camera = self.camera_initialization(resolution_x, resolution_y)

        
        # batch size
        batch_size = self.batch_size
        iteration = int(self.number_photon/batch_size)
        # per bacth size
        # d_measure,count = self.measurement_initialization(batch_size)

        # 5.7.1 initialize the media
        self.meida_initialization()

        # # 5.7.2 initialize the photons
        P_vectors, D_vectors, Albedo = self.photons_initialization(self.number_photon)
        num_deserter = torch.tensor([0]).type(self.dtype) 


        
        for n in tqdm(range(1,iteration),ncols=80):
            
            # initalize the anchor
            # d_measure,count = self.measurement_initialization(batch_size)

            # d_max = d_measure[0,:,:]
            # d_avg = d_measure[2,:,:]
            # count_number = count[0,:,:]
            # number_t = count[2,:,:]
            
            stokes = self.stokes
            Scattering_matrix = self.Scattering_matrix
            Palf_norm =  self.Palf_norm

            p_vectors=P_vectors[n*batch_size:(n+1)*batch_size,:]
            d_vectors=D_vectors[n*batch_size:(n+1)*batch_size,:]
            albedo=Albedo[n*batch_size:(n+1)*batch_size,:]

            print("No.{} bacth of photons, launch ! !".format(n))
            # 5.7.3 initialize the position, polarized state, albedo and step
            # for each photons 
            ift = False
            depth_measure=measurement()
            first_move = position(p_vectors, d_vectors)
            p_vectors = first_move.random_walks(self.miu_t)
            
            photons_states = torch.ones(batch_size,1).type(self.dtype) 
            alive_mask = (photons_states==1)
            
            # 5.7.4 judgement 
            # judgement of position: check if it still in the "space"
            # judgement of albedo: also we need to check for the intensity
            while torch.sum(photons_states)>0:

                # photons_states =self.mask_det_position(p_vectors,photons_states)
                # zero_mask = (photons_states==0)
                # zeros_indices = zero_mask.nonzero(as_tuple=True)
                
                mie_scattering = photons_event(Scattering_matrix,stokes,Palf_norm,self.nalfa,self.nbeta,photons_states.size()[0])
                alfa,beta,stokes = mie_scattering.mie_scattering_particles(photons_states.size()[0],alive_mask)

                stokes,d_vectors,flag_C=self.update_polarized_states(alfa,beta,p_vectors,d_vectors,stokes)

                move = position(p_vectors, d_vectors)
                p_vectors = move.random_walks(self.miu_t)


                try:
                    photons_states,p_vectors, d_vectors, albedo, stokes, mask_a, mask_d = self.albedo_det_death(photons_states, p_vectors, d_vectors, albedo, stokes)
                    num_deserter = num_deserter + mask_d.sum()
                    print("1: alive photons number is {}".format(photons_states.size()))
                    print("1: alive photons number is {}".format(p_vectors.size()))

                    if d_vectors.ndim==3:
                        print("existing some errors")
                        break
                    # photons_states[0,0]=0
                    # photons_states[10,0]=0
                    # photons_states[21,0]=0

                    photons_states,p_vectors, d_vectors, albedo, stokes, mask_p, mask_d, t_camera, b_camera= self.position_det_death(photons_states, p_vectors, d_vectors, albedo, stokes, t_camera, b_camera)
                    num_deserter = num_deserter + mask_d.sum()
                    print("2: alive photons number is {}".format(photons_states.size()))
                    print("2: alive photons number is {}".format(p_vectors.size()))
                    gen_d = mymath()
                    alive_mask = gen_d.gen_death_mask(mask_a, mask_p)
                    Palf_norm = Palf_norm[alive_mask.squeeze(dim=1),:]
                except(RuntimeError):
                    break
                # print(p_vectors)

            # while self.mask_det_albedo(p_vectors,albedo) & self.mask_det_position(p_vectors):
            #     print("lots of thing to do!")
                
            # while self.det_albedo(p_vectors,albedo) & self.det_position(p_vectors):
            #         # 5.7.5 mie scattering
            #         mie_scattering = photons_event(self.Scattering_matrix,stokes,self.Palf_norm,self.nalfa,self.nbeta)
            #         alfa,beta,stokes = mie_scattering.mie_scattering_particles()
            #         # print(alfa,beta)
            #         # 5.7.6 update the out direction
            #         # print(d_vectors)
            #         # d_vectors=self.out_direction(alfa,beta,p_vectors,d_vectors)
            #         # print(d_vectors)

            #         # 5.7.7 update the direction and polarized sates
            #         # print("S_vector is {}".format(stokes))
            #         # nan_probe =[np.isnan(i) for i in stokes]
            #         # print("nan probe is {}".format(nan_probe))
            #         stokes,d_vectors,flag_C=self.update_polarized_states(alfa,beta,p_vectors,d_vectors,stokes)
            #         # print("updated S_vector is {}".format(stokes))
            #         # nan_probe =[np.isnan(i) for i in stokes]
            #         # print("updated nan probe is {}".format(nan_probe))

            #         if flag_C:
            #                 # 5.7.8 change the position based on the out-direction
            #                 move = position(p_vectors, d_vectors)
            #                 p_vectors = move.random_walk(self.miu_t)
            #                 d_max = depth_measure.record_depth(d_max,p_vectors[2])
            #                 if p_vectors[2]>self.d:
            #                     d_max = self.d
            #                     ift = True
            #                     # d_max = depth_measure.record_depth(d_max,p_vectors[2])
            #                     break
            #                 # 5.7.9 update the albedo
            #                 albedo = self.update_albedo(albedo, self.miu_s, self.miu_a)
            #         else:
            #             if p_vectors[2]>self.d:
            #                 number_t = number_t + 1 
            #                 d_max = self.d
            #                 # ift = True
            #                 break

            #             d_max =  0
            #             count = count + 1
            #             break
            # # 5.7.10 
            # # jump out of the while loop
            # # collect the image by the so-called stokes-camera 
            # # result 
            # # p_vectors, d_vectors, stokes, albedo, d_max, count_number, number_t, ift
            # if ift:
            #     t_camera = self.last_action_transmitted_photon(p_vectors, d_vectors, stokes, albedo, t_camera)
            # else:
            #     stokes_camera = self.last_action_photon(p_vectors, d_vectors, stokes, albedo, stokes_camera)
            #     d_measure[1] = d_measure[1]+ d_max
            #     count[1] = count[1]+ count_number
            #     count[3] = count[3]+ number_t
            #     # print(np.sum(stokes_camera)) 
                 
        return t_camera, b_camera, num_deserter
    
    def ParTrans(self, resolution_x, resolution_y, event):
        # 5.7 the whole procedure
        # 5.7.0 initialize the camera
        stokes_camera = self.camera_initialization(resolution_x, resolution_y)
        t_camera = self.camera_initialization(resolution_x, resolution_y)
        # 5.7.1 initialize the measurement
       
        d_measure,count = self.measurement_initialization()
        # d_measure = 0
        # d_measure_sum = 0
        
        # 5.7.2 initialize the media
        self.meida_initialization()

        # 5.7.3 initialize the photons
        p_vectors, d_vectors, albedo = self.photons_initialization(self.number_photon)

        #  mp.cpu_count() 
        #  To know how many process i have 
        
        # num_process = mp.cpu_count()-1
        num_process = 20
        
        with Pool(processes=num_process) as pool:  # Change the number of processes as needed
            if event.startswith('sphere'):
                results = pool.starmap(self.process_photon, [(self.stokes, self.Palf_norm, n, self.nalfa, self.nbeta, p_vectors[n,:], d_vectors[n,:], albedo, d_measure[0], d_measure[2], count[0],count[2]) for n in range(int(self.number_photon))])
            elif event.startswith('cylinder'):
                results = pool.starmap(self.cylinder_process_photon, [(self.stokes, self.Palf_norm, n, self.nalfa, self.nbeta, p_vectors[n-1,:], d_vectors[n-1,:], albedo, d_measure[0], d_measure[2], count[0],count[2],) for n in range(int(self.number_photon))])

        for idx_photon,result in enumerate(results):
             if result[7]:
                # print("No.{}, trans_stokes is {}".format(idx_photon, result[2]))
                t_camera = self.last_action_transmitted_photon(result[0], result[1], result[2], result[3], t_camera)
                # print("result is {}".format(t_camera))
             else:
                stokes_camera = self.last_action_photon(result[0], result[1], result[2], result[3], stokes_camera)
                d_measure[1] = d_measure[1]+ result[4]
                count[1] = count[1]+ result[5]
                count[3] = count[3]+ result[6]
                print("No. {} photon, penetration depth is {} ".format(idx_photon, d_measure[1]))

        print("Effective photons number is {}".format(self.number_photon-count[1]-count[3]))
        d_max = d_measure[1]/(self.number_photon-count[1]-count[3])

        return t_camera,stokes_camera,d_max,(self.number_photon-count[1])

    # def process_photon(self, stokes, Palf_norm, photon_index, nalfa, nbeta, p_vectors, d_vectors, albedo, d_max, d_avg, count,number_t):
    #     print("No.{} photon, launch ! !".format(photon_index))
    #     # 5.7.3 initialize the position, polarized state, albedo and step
    #     # for each photon
    #     # p_vectors, d_vectors, albedo = self.photons_initialization(self.number_photon)
    #     # initialize the position: first movement
    #     ift = False
    #     depth_measure=measurement()
    #     first_move = position(p_vectors, d_vectors)
    #     p_vectors = first_move.random_walk(self.miu_t)
    #     # 5.7.4 judgement
    #     # judgement of position: check if it still in the "space"
    #     # judgement of albedo: also we need to check for the intensity
    #     while self.det_albedo(p_vectors, albedo) & self.det_position(p_vectors):
    #         # 5.7.5 mie scattering
    #         #start_time = time.time()
    #         mie_scattering = photon_event(self.Scattering_matrix, stokes, Palf_norm, nalfa, nbeta)
    #         alfa, beta, stokes = mie_scattering.mie_scattering_particle()
    #         #end_time = time.time()
    #         #print("time for define the photon event is{}".format(end_time-start_time))
    #         # 5.7.6 update the out direction
    #         # d_vectors=self.out_direction(alfa,beta,p_vectors,d_vectors)
    #         # 5.7.7 update the direction and polarized sates
    #         stokes, d_vectors, flag_C = self.update_polarized_states(alfa, beta, p_vectors, d_vectors, stokes)
    #         if flag_C:
    #             # 5.7.8 change the position based on the out-direction
    #             move = position(p_vectors, d_vectors)
    #             p_vectors = move.random_walk(self.miu_t)
    #             d_max = depth_measure.record_depth(d_max,p_vectors[2])
    #             if p_vectors[2]>self.d:
    #                 d_max = self.d
    #                 ift = True
    #                 # d_max = depth_measure.record_depth(d_max,p_vectors[2])
    #                 break
    #             # 5.7.9 update the albedo
    #             albedo = self.update_albedo(albedo, self.miu_s, self.miu_a)
    #         else:
    #             if p_vectors[2]>self.d:
    #                 number_t = number_t + 1 
    #                 d_max = self.d
    #                 # ift = True
    #                 break

    #             d_max =  0
    #             count = count + 1
    #             break

            
    #         # if p_vectors[2]>self.d:
    #         #    number_t = number_t + 1  
    #         #    d_max =self.d

    #     return p_vectors, d_vectors, stokes, albedo, d_max, count, number_t, ift

    def Trans_test(self,resolution_x,resolution_y,Palf_norm, nzeta, nphi, p_vectors, d_vectors,r,k,an1,an2,bn1,bn2,stokes_in,qsca,l,vf,P):
        # 5.7 the whole procedure
        # 5.7.0 initialize the camera
        stokes_camera=self.camera_initialization(resolution_x,resolution_y)
        t_camera = self.camera_initialization(resolution_x, resolution_y)
        # 5.7.1 initialize the measurement
       
        d_measure,count = self.measurement_initialization()
        # d_measure = 0
        # d_measure_sum = 0
        

        # 5.7.1 initialize the media
        self.meida_initialization()
        # 5.7.2 initialize the photons
        ift = False
        # depth_measure=measurement()
        # first_move = position(p_vectors, d_vectors)

        # p_vectors = first_move.random_walk(mie_cylinder_miut[n2])

        
        for n in tqdm(range(1,int(self.number_photon)),ncols=40,position=0, leave=True):
            stokes=self.stokes
            # print("No.{} photon, launch ! !".format(n))
            #  5.7.3 initialize the position, polarized state, albedo and step
            # for each photons 
            p_vectors, d_vectors,albedo = self.photon_initialization()
            # initialize the position: first movement
            first_scattering = photon_event(self.Scattering_matrix, stokes, Palf_norm, nzeta, nphi)
                
                # diameter=2*r
                # ca = 0.01
                # n_zeta = n1
                # n_theta = n2
                
            angle_cal = mymath()
            A = angle_cal.cal_angle_differece(d_vectors,l)
            miu_t=first_scattering.mie_cylinder_miut(2*r,vf,A,nzeta,nphi,k,an1,an2,bn1,bn2,stokes_in,qsca)
            first_move = position(p_vectors, d_vectors)

            p_vectors = first_move.random_walk(np.real(miu_t))
            # 5.7.4 judgement 
            # judgement of position: check if it still in the "space"
            # judgement of albedo: also we need to check for the intensity
            while self.det_albedo(p_vectors,albedo) & self.det_position(p_vectors):
                mie_scattering = photon_event(self.Scattering_matrix, stokes, Palf_norm, nzeta, nphi)
                    
                theta, phi, stokes = mie_scattering.mie_scattering_cylinder(P,d_vectors,l,nzeta,nphi,an1,an2,bn1,bn2)
                # alfa, beta, stokes = mie_scattering.mie_scattering_particle()
                stokes, d_vectors, flag_C = self.update_polarized_states(theta, phi, p_vectors, d_vectors, stokes)
                # print("updated stokes vector is {}".format(stokes))
                # print("updated d_vector is {}".format(d_vectors))
                if flag_C:
                    # 5.7.8 change the position based on the out-direction
                    move = position(p_vectors, d_vectors)
                    p_vectors = move.random_walk(miu_t)
                    #    print("continues to transfer")
                    #    print("updated p_vector is {}".format(p_vectors))
                    #    d_max = depth_measure.record_depth(d_max,p_vectors[2])
                    if p_vectors[2]>self.d:
                        #    d_max = self.d
                        ift = True
                        # d_max = depth_measure.record_depth(d_max,p_vectors[2])
                        break
                    # 5.7.9 update the albedo
                    #    albedo = self.update_albedo(albedo, self.miu_s[0], self.miu_a[0])

                    # albedo = self.update_albedo_cylinder(albedo, self.miu_s, self.miu_a,miu_t)
                else:
                    if p_vectors[2]>self.d:
                        number_t = number_t + 1 
                        # d_max = self.d
                        # ift = True
                        break

                    # d_max =  0
                    count = count + 1
                    break

            
            # if p_vectors[2]>self.d:
            #    number_t = number_t + 1  
            #    d_max =self.d

            # return p_vectors, d_vectors, stokes, albedo, d_max, count, number_t, ift  
            if ift:
                t_camera = self.last_action_transmitted_photon(p_vectors, d_vectors, stokes, albedo, t_camera)  
            else:    
                stokes_camera = self.last_action_photon(p_vectors, d_vectors, stokes, albedo, stokes_camera)

        return t_camera,stokes_camera

    def cylinder_process_photon(self, stokes, Palf_norm, photon_index, nzeta, nphi, p_vectors, d_vectors, albedo, d_measure, d_max, count, number_t, r, vf, k, an1, an2, bn1, bn2, qsca2, l, P):
        # Initialize the photon properties
        ift = False
        depth_measure = measurement()
        first_scattering = photon_event(self.Scattering_matrix, stokes, Palf_norm, nzeta, nphi)
        angle_cal = mymath()
        A = angle_cal.cal_angle_differece(d_vectors, l)
        miu_t = first_scattering.mie_cylinder_miut(2 * r, vf, A, nzeta, nphi, k, an1, an2, bn1, bn2, stokes, qsca2)
        # print("calculate miu_t")
        first_move = position(p_vectors, d_vectors)
        p_vectors = first_move.random_walk(np.real(miu_t))
        # d_max = 0
        # count = 0
        # number_t = 0 
        # The main loop for photon processing
        while self.det_albedo(p_vectors, albedo) & self.det_position(p_vectors):
            mie_scattering = photon_event(self.Scattering_matrix, stokes, Palf_norm, nzeta, nphi)
            theta, phi, stokes = mie_scattering.mie_scattering_cylinder(P, d_vectors, l, nzeta, nphi, an1, an2, bn1, bn2)
            stokes, d_vectors, flag_C = self.update_polarized_states(theta, phi, p_vectors, d_vectors, stokes)
            # print("upadated polarized states")
            if flag_C:
                move = position(p_vectors, d_vectors)
                A = angle_cal.cal_angle_differece(d_vectors, l)
                miu_t = first_scattering.mie_cylinder_miut(2 * r, vf, A, nzeta, nphi, k, an1, an2, bn1, bn2, stokes, qsca2)
                p_vectors = move.random_walk(np.real(miu_t))
                d_max = depth_measure.record_depth(d_max, p_vectors[2])
                if p_vectors[2] > self.d:
                    d_max = self.d
                    ift = True
                    break
                albedo = self.update_albedo_cylinder(albedo, self.miu_s, self.miu_a,miu_t)
            else:
                if p_vectors[2] > self.d:
                    number_t = number_t + 1 
                    d_max = self.d
                    break
                d_max = 0
                count = count + 1
                break
        
        # bar()
        return p_vectors, d_vectors, stokes, albedo, d_max, count, number_t, ift

    def Partrans_cylinder(self, resolution_x, resolution_y, Palf_norm, nzeta, nphi, r, vf, k, an1, an2, bn1, bn2, stokes_in, qsca, l, P):
        # Initialization
        stokes_camera = self.camera_initialization(resolution_x, resolution_y)
        t_camera = self.camera_initialization(resolution_x, resolution_y)
        d_measure, count = self.measurement_initialization()
        self.meida_initialization()

        # 5.7.3 initialize the photons
        p_vectors, d_vectors, albedo = self.photons_initialization(self.number_photon)


        with Pool(processes=18) as p:
            # results = list(tqdm(p.starmap(self.cylinder_process_photon, [(stokes_in, Palf_norm, n, nzeta, nphi, p_vectors[n-1,:], d_vectors[n-1,:], albedo, d_measure[0], d_measure[2], count[0],count[2], r, vf, k, an1, an2, bn1, bn2, qsca, l, P) for n in range(1, int(self.number_photon))])),total=self.number_photon)
            # with alive_bar(self.number_photon) as bar:
            results = p.starmap(self.cylinder_process_photon, [(stokes_in, Palf_norm, n, nzeta, nphi, p_vectors[n-1,:], d_vectors[n-1,:], albedo, d_measure[0], d_measure[2], count[0],count[2], r, vf, k, an1, an2, bn1, bn2, qsca, l, P) for n in range(1, int(self.number_photon))])

        # Post-process the results
        for result in results:
            p_vectors, d_vectors, stokes, albedo, d_max, count, number_t, ift = result
            if ift:
                t_camera = self.last_action_transmitted_photon(p_vectors, d_vectors, stokes, albedo, t_camera)
            else:
                stokes_camera = self.last_action_photon(p_vectors, d_vectors, stokes, albedo, stokes_camera)
        
        

            d_measure[1] = d_measure[1]+ result[4]
            count[1] = count[1]+ result[5]
            count[3] = count[3]+ result[6]
            # print("No. {} photon, penetration depth is {} ".format(idx_photon, d_measure[1]))

        # print("Effective photons number is {}".format(self.number_photon-count[1]-count[3]))
        d_max = d_measure[1]/(self.number_photon-count[1]-count[3])

        return t_camera,stokes_camera,d_max,(self.number_photon-count[1])
        # return t_camera, stokes_camera

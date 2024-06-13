# User instruction

# 1. Author: Luna (Yuxuan Mao)

# 2. Date: from 11/09 to ...

# 3. Function: class of Mueller formalism
                # calculation for mueller matrix (polarized light)
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
from operator import methodcaller

class polarized_light:
    # 5 generate the stokes vector of polarized_light
    def __init__(self,I,e0x,e0y,retardance):
        self.I = I
        self.e0x = e0x
        self.e0y = e0y
        self.retardance=retardance
    def LHP(self):
        # linear horizontally polarized light
        return self.I*np.array([1,1,0,0])
    def LVP(self):
        # linear vertically polarized light
        return self.I*np.array([1,-1,0,0])
    def Lp45(self):
        # linear +45 polarized light
        return self.I*np.array([1,0,1,0])
    def Lm45(self):
        # linear -45 polarized light
        return self.I*np.array([1,0,-1,0])    
    def RCP(self):
        # right circularly polarized light
        return self.I*np.array([1,0,0,1])
    def LCP(self):
        # left circularly polarized light
        return self.I*np.array([1,0,0,-1])    
    def custom(self):
        # Custom polarized light
        # "Polarized Light--Second edition--Dennis Goldstein
        # Chapter 3, page 35-43 
        s0=self.e0x**2+self.e0y**2
        s1=self.e0x**2-self.e0y**2
        s2=2*self.e0x*self.e0y*math.cos(self.retardance)
        s3=2*self.e0x*self.e0y*math.sin(self.retardance)
        return np.array(s0,s1,s2,s3)

class polarized_mueller_matrix(polarized_light):
    # 6 generate polarized_light and calculate everything with mueller formalism
    def __init__(self,I,e0x,e0y,retardance):
        
        # 6.1.0* super class
        super().__init__(I,e0x,e0y,retardance)

        # 6.1.1 scattering coefficient
        # self.miu_s  = miu_s
        # 6.1.2 absorption coefficient
        # self.miu_a  = miu_a

        # 6.1.3 Extinction coefficient
        # The attenuation of light due to the absorption and scattering 
        # may be characterized by the extinction coefficient
        # Source:
        # Polarized Light in Biomedical Imaging and Sensing Clinical and Preclinical Applications 
        # Jessica C. Ramella-Roman, Tatiana Novikova
        # page 107

        # self.miu_e  = miu_s + miu_a
        
        # 6.1.4 miu_sk, from Albert's code
        # self.miu_sk = self.miu_s/self.miu_e

    def gpl(self,mode):
        # 6.2.1  generate_polarized_light(self,mode)
        # function: 
        # Target:   
        # Input:    
        # output:

        # 6.2.2 Source: 
        # "Polarized light, second edition" 
        # Dennis Goldstein 
        # chapter 4, page 38

        # stokes = np.zeros((4,1),dtype=np.complex_)

        light1 = polarized_light(self.I,self.e0x,self.e0y,self.retardance)
        # TODO 
        # if the mode is not custom, then 
        # e0x,e0y,retardance could be any number XD
        handle_func = methodcaller(mode)
        stokes = handle_func(self)
        # Luna's note:
        # if you use: 
        # stokes = light1.mode()
        # you will get a ERROR: 'polarized_light' object has no attribute 'mode'
        # since you can't find any mode in PMM
        # so you have to use super-class in PYTHON
        # it is useful!!!!!
        return stokes
    
    def mueller_matrix_multiply():
        pass

    def mueller_matrix_rotation():
        pass

    def cal_intensity():
        pass
    


    def skyrmions():
        pass

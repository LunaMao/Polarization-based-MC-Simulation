# User instruction

# 1. Author: Luna (Yuxuan Mao)

# 2. Date: from 11/08 to ...

# 3. Function: class of Mie scattering
#    including:
#    3.1 mie_an_bn : to calculate the n-order Mie coefficients

# 4. Copyright
# 4.1 Optical Metrology & Image Processing Laboratory (MIP-OptiLab)
# Sincere and hearty thanks to: Prof.Juan Campos, Prof.Angel Lizana, Irene Estévez
# Respect to the original author of Matlab version: Albert Van Eeckhout Alsinet
# Extremely grateful to Ivan Montes
# LOVE ALL OF YOU!!!!!

# Muchas gracias, Barcelona, y mi universidad, Universitat Autònoma de Barcelona.
# A mi me gusta esta ciudad.

# 4.2 CSC funding
# Supported by the China Scholarship Council
# It is my honor to benefit from my motherland


import math
import cupy as np
# import numpy as np
from scipy.special import spherical_jn
from scipy.special import spherical_yn
from scipy.special import hankel1 
from scipy.special import riccati_jn
from scipy.special import riccati_yn
# import treams
from utility import mymath
# from utility import utility

# 5.1 Mie_scattering
class Mie_scattering:
    def __init__(self, m, k, vf, r):
        self.m = m
        self.x = k * r
        self.vf = vf
        self.r = r
        # self.nbeta = nbeta

    def Sphere_mie_an_bn(self):
        # 5.1.1 function: 
        # to compute n-order Mie coefficients, n, a_n, b_n
        # complex refractive index m=m'+im"

        # TODO: refine this part
        # input: self
        # maybe m and x from self

        # output:
        # n, an and bn
        # for the calculate of scattering coefficient

        # 5.1.2 Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 99 -- 4.3.3 Scattering Coefficients 
        # Page 477 -- appendix A

        # 5.1.3.Detailed note
        # 5.1.3.1 Set the end of loop: 
        # 5.1.3.1.1 n-stop 
        nmax=round(2+self.x+4*math.pow(self.x,1/3))
        # 5.1.3.1.2 create array n, from 1 to n-stop
        n=range(1,nmax+1)
        # for gradient calculation
        nn=range(0,nmax)
        # 5.1.3.2 abberation for different variables 
        # 5.1.3.2.1 mx
        mx=self.m * self.x
        # 5.1.3.2.2 m^2
        m2=self.m * self.m

        # *5.1.4. Additional note for complex functions!!

        # 5.1.4.1 First kind spherical Bessel -- jn
        # --> spherical_jn (n, z, derivative=False)

        # Parameters:
        #     n: int, array_like
        #         Order of the Bessel function (n >= 0).

        #     z: complex or float, array_like
        #         Argument of the Bessel function.

        #     derivative: bool, optional
        #         If True, the value of the derivative (rather than the function itself) is returned.

        # Returns: jn ndarray
        
        jn_x=spherical_jn(n,self.x)
        jn_mx=spherical_jn(n,mx)
        yn_x=spherical_yn(n,self.x)

        jn_mx = np.array(jn_mx)
        jn_x = np.array(jn_x)
        yn_x = np.array(yn_x)
        # 5.1.4.2 The first kind of Hankel function -- h_n_1(x)
        # also, third Kind of Bessel Functions
        # first kind: = jv(x)+i*Yv(x) x>0
        # *second kind: = jv(x)-i*Yv(x) x>0
        
        # --> hankel1(v, z, out=None) 

        # Parameters:
        #     v: array_like
        #         Order (float).

        #     z: array_like
        #         Argument (float or complex).

        #     out: ndarray, optional
        #         Optional output array for the function values

        # Returns:  scalar or ndarray
        #         Values of the Hankel function of the first kind.

        hn1_x=[complex(jn_x[index-1],-yn_x[index-1]) for index in n]
        hn1_x = np.array(hn1_x)
        jn_x_0=spherical_jn(nn,self.x)
        yn_x_0=spherical_yn(nn,self.x)
        hn1_x_0=[complex(jn_x_0[index-1],yn_x_0[index-1]) for index in n]
        hn1_x_0 = np.array(hn1_x_0)
        # the original function is wrong???
        # hn1_x=hankel1(n,self.x)

        # 5.1.4.3 Riccati_Bessel of the first kind -- [x*J_n(x)]
        # --> scipy.special.riccati_jn(n, x)
        # Parameters:
        #     n: int
        #         Maximum order of function to compute
        #        !!!!! Maxxxxx order !!!!!
        #     x: float
        #         Argument at which to evaluate

        # Returns:
        #     jn: ndarray
        #         Value of j0(x), …, jn(x)
        #     jnp: ndarray
        #     First derivative j0’(x), …, jn’(x)
        
        xjn_x, xjn_x_g   = riccati_jn(nmax,self.x)
        xjn_x=xjn_x[1:]
        xjn_x = np.array(xjn_x)

        xjn_x_g=xjn_x_g[1:]
        xjn_x_g = np.array(xjn_x_g)

        mxjn_mx, mxjn_mx_g = riccati_jn(nmax,mx)

        mxjn_mx_g=mxjn_mx_g[1:]
        mxjn_mx_g = np.array(mxjn_mx_g)


        # 5.1.4.4 Riccati_Henkel [x*h_n_1(x)] (type 3)
        # there are four types of Riccati–Bessel functions
        # Sn and Cn refers to riccati_jn(n, x) and riccati_yn(n, x), respectively
        # Xin = Sn - i*Cn 
        # Zetan = Sn + i*Cn
        # https://en.wikipedia.org/wiki/Bessel_function
        # "Riccati–Bessel functions"

        # 5.1.4.4.1 Albert's method
        # Xin_x_g = self.x*np.array(hn1_x_0)-n*np.array(hn1_x)
        # 5.1.4.4.2 Luna's method
        xyn_x, xyn_x_g   = riccati_yn(nmax,self.x)
        xyn_x_g=xyn_x_g[1:]
        xyn_x_g = np.array(xyn_x_g)

        # gradient = real_g + i*img_g

        Xin_x_g= [complex(xjn_x_g[index-1], xyn_x_g[index-1]) for index in n]
        Xin_x_g=np.array(Xin_x_g)
        # 5.1.5. Calculation of Coeficient
        # 5.1.5.1 Equation of an
        # 5.1.5.1.1 numerator 

        # num_an = m2*jn_mx*xjn_x_g-jn_x*mxjn_mx_g

        num_an = m2*jn_mx*xjn_x_g-jn_x*mxjn_mx_g
        # 5.1.5.1.2 denominator
        denum_an = m2*jn_mx*Xin_x_g-hn1_x*mxjn_mx_g
        
        self.an=num_an/denum_an

        # 5.1.5.2 Equation of bn
        # numerator 
        # 5.1.5.2.1 numerator 
        num_bn=jn_mx*xjn_x_g-jn_x*mxjn_mx_g      
        # 5.1.5.2.2 denominator
        denum_bn=jn_mx*Xin_x_g-hn1_x*mxjn_mx_g     
        
        self.bn=num_bn/denum_bn
        self.n=n
        return self.n, self.an, self.bn
    
    def Sphere_coefficient(self):
        # 5.2.1 function: 
        # to compute coefficients for absorption and scattering 
        # miu_a, miu_s
        # complex refractive index m=m'+im"
        # an and bn for n=1 to nmax

        # TODO: refine this part
        # input: self

        # output:
        # miu_a, miu_s

        # 5.2.2 Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # 5.2.2.1
        # efficiencies for extiniction (qext), scattering (qsca), absorption (qabs), backscattering (qb)
        # 5.2.2.1.1 note 1
        # page 72 
        # eg. qext=cext/G
        # G is the particle cross-sectional area projected 
        # onto a plane perpendicular to the incident beam
        # G = pi*a^2 for a sphere of radius a


        # 5.2.2.1.1 note 2
        # page 136
        # qabs=qext-qsca
        # good approximation: eq (5.10) and eq.(5.11)

        # 5.2.2.1.3 note 3
        # page 204 
        # efficiencies for infinite cylinder

        # 5.2.2.1.3 note 4
        # page 103
        # eq (4.60) and eq.(4.61)
        # csca and cext
        # q=c/G =(2pi/k^2)/(pi*r^2)=2/(k*r)^2=2/x^2
        n=np.array([self.n])

        qext=2*np.sum((2*n+1)*np.real(self.an+self.bn))/self.x**2
        qsca=2*np.sum((2*n+1)*(abs(self.an)**2+abs(self.bn)**2))/self.x**2
        qabs=qext-qsca
        
        self.miu_a=3*qabs*self.vf*10000/(4*self.r)
        self.miu_s=3*qsca*self.vf*10000/(4*self.r)

        return self.miu_s,self.miu_a
    
    def sphere_mie_treams(self,k0s,index,lmax, radius):
        # k0s = 2*np.pi*np.linspace(1/700,1/300,200)
        # index = 16 + 0.5j
        materials = [treams.Material(index),treams.Material()]
        lmax=4
        radius =1000
        # TODO
        sphere = [treams.TMatrix.sphere(lmax,k0,radius, materials) for k0 in k0s]
        # spheres = [treams.TMatrix.sphere(lmax,k0,radius,materials) for k0 in k0s]
        swb_lmax1=treams.SphericalWaveBasis.default(1)
        # spheres_lmax1 = sphere[swb_lmax1]
        # xs_sca_lmax1 = np.array(spheres_lmax1.xs_sca_avg)/(np.pi*radius**2) 
        # xs_ext_lmax1 = np.array(spheres_lmax1.xs_ext_avg)/(np.pi*radius**2)
        # return xs_sca_lmax1, xs_ext_lmax1


        spheres_lmax1 = [tm[swb_lmax1] for tm in sphere] 
        xs_sca_lmax1 = [np.array(tm.xs_sca_avg)/(np.pi*radius**2) for tm in spheres_lmax1]
        xs_ext_lmax1 = [np.array(tm.xs_ext_avg)/(np.pi*radius**2) for tm in spheres_lmax1]
        return np.array(xs_sca_lmax1), np.array(xs_ext_lmax1)

    def cylinder_mie_an_bn_c1(self,zeta):
        
        # Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 198 -- eq 8.29 

        #  k wave number
        #  a radius
        #  x = k * a
        
        xi = self.x *np.sin(zeta) 
        if (self.m**2-np.cos(zeta)**2)>=0:
            eta = self.x *np.sqrt(np.abs(self.m**2-np.cos(zeta)**2))
        else:
            eta = self.x *1j*np.sqrt(np.abs(self.m**2-np.cos(zeta)**2))
        
        nmax=round(2+self.x+4*math.pow(self.x,1/3))
        # n-stop
        # n=range(1,nmax+1)
        n=range(0,nmax)
        jn_eta=spherical_jn(n,eta)

        jn_xi=spherical_jn(n,xi)
        yn_xi=spherical_yn(n,xi)
        yn_xi_g=spherical_yn(n,xi,True)
        hn1_xi=[complex(jn_xi[index],-yn_xi[index]) for index in n]
        # Dn 
        Dn = n*np.cos(xi)*eta*jn_eta*hn1_xi*(xi**2/eta**2-1)
        # Cn 
        Cn = n*np.cos(xi)*eta*jn_eta*jn_xi*(xi**2/eta**2-1)
        # Bn
        jn_eta_g=spherical_jn(n,eta,True)
        jn_xi_g=spherical_jn(n,xi,True)

        Bn = xi*(self.m**2*xi*jn_eta_g*jn_xi-zeta*jn_eta*jn_xi_g)
        # Vn
        hn1_xi_g=[complex(jn_xi_g[index],-yn_xi_g[index]) for index in n]
        Vn = xi*(self.m**2*xi*jn_eta_g*hn1_xi-eta*jn_eta*hn1_xi_g)
        # Wn
        Wn = 1j* xi* (eta*jn_eta*hn1_xi_g-xi*jn_eta_g*hn1_xi)

        cc =mymath()
        # complex calculator
        # an for class 1 
        self.an1 = np.zeros((Wn.shape[0],1),dtype ='complex_')
        # wv=cc.complex_multi(Wn,Vn)
        self.an1 = (Cn*Vn - Bn*Dn)/(Wn*Vn+1j*Dn**2)
        # bn for class 1 
        self.bn1 = np.zeros((Wn.shape[0],1),dtype ='complex_')
        self.bn1 = (Wn*Bn + 1j*Dn*Cn)/(Wn*Vn+1j*Dn**2)

        return nmax,self.an1, self.bn1
    
    def cylinder_parallel_coefficient(self,h):
        sq_term = np.sum(((np.abs(self.bn1))**2+(np.abs(self.an1))**2))
        qsca1 = 2/self.x*(2*sq_term-2*np.abs(self.an1[0])**2-np.abs(self.bn1[0])**2)
        qext1 = 2/self.x*np.real(2*np.sum(self.bn1)-self.bn1[0])
        qabs1=qext1-qsca1
        return qsca1,qext1,qabs1
    
    def cylinder_parallel_efficiencies(self,h):
        # height of cylinder: h
        # qsca1 = 2/self.x*(np.sum(np.abs(self.an1)**2+np.abs(self.bn1)**2)-self.an1[0]**2)
        # qext1 = np.real(2/self.x*np.sum(np.abs(self.bn1)**2))
        # qabs1=qext1-qsca1
        qsca1,qext1,qabs1 = self.cylinder_parallel_coefficient(h)
        c = np.pi*self.r**2
        self.miu_s1=qsca1*self.vf*h*10000
        self.miu_a1=qabs1*self.vf*h*10000

        return self.miu_s1,self.miu_a1
    
    def cylinder_mie_an_bn_c2(self,zeta):
        
        # Source: 
        # "Absorption and Scattering of Light by Small Particles" 
        # CRAIG F. BOHREN & DONALD R. HUFFMAN (1983)
        # page 199 -- eq 8.31 

        #  k wave number
        #  a radius
        #  x = k * a
        
        xi = self.x *np.sin(zeta) 
        if (self.m**2-np.cos(zeta)**2)>=0:
            eta = self.x *np.sqrt(np.abs(self.m**2-np.cos(zeta)**2))
        else:
            eta = self.x *1j*np.sqrt(np.abs(self.m**2-np.cos(zeta)**2))
        # eta = self.x *np.sqrt(np.abs(self.m**2-np.cos(zeta)**2))
        
        nmax=round(2+self.x+4*math.pow(self.x,1/3))
        # n-stop
        # n=range(1,nmax+1)
        n=range(0,nmax)
        # n=range(1,nmax+1)
        # nnn=range(1,nmax)
        jn_eta=spherical_jn(n,eta)

        jn_xi=spherical_jn(n,xi)
        yn_xi=spherical_yn(n,xi)
        yn_xi_g=spherical_yn(n,xi,True)
        hn1_xi=[complex(jn_xi[index],-yn_xi[index]) for index in n]
        # Dn 
        Dn = n*np.cos(xi)*eta*jn_eta*hn1_xi*(xi**2/eta**2-1)
        # Cn 
        Cn = n*np.cos(xi)*eta*jn_eta*jn_xi*(xi**2/eta**2-1)
        # An
        jn_eta_g=spherical_jn(n,eta,True)
        jn_xi_g=spherical_jn(n,xi,True)
        An = 1j*xi*(xi*jn_eta_g*jn_xi-eta*jn_eta*jn_xi_g)
        # Vn
        hn1_xi_g=[complex(jn_xi_g[index],-yn_xi_g[index]) for index in n]
        Vn = xi*(self.m**2*xi*jn_eta_g*hn1_xi-eta*jn_eta*hn1_xi_g)
        # Wn
        Wn = 1j* xi* (eta*jn_eta*hn1_xi_g-xi*jn_eta_g*hn1_xi)

        # an for class 1 
        self.an2 = np.zeros((Wn.shape[0],1),dtype ='complex_')
        self.an2 = -(An*Vn - 1j*Cn*Dn)/(Wn*Vn+1j*Dn**2)
        # bn for class 1 
        self.bn2 = np.zeros((Wn.shape[0],1),dtype ='complex_')
        self.bn2 = -1j*(Cn*Wn + An*Dn)/(Wn*Vn+1j*Dn**2)

        return nmax,self.an2, self.bn2

    def cylinder_perpendicular_coefficient(self,h):
        sq_term = np.sum(((np.abs(self.bn2))**2+(np.abs(self.an2))**2))
        qsca2 = 2/self.x*(2*sq_term-np.abs(self.an2[0])**2-2*np.abs(self.bn2[0])**2)
        qext2 = 2/self.x*np.real(2*np.sum(self.an2)-self.an2[0])
        qabs2=qext2-qsca2
        # print(qsca2)
        return qsca2,qext2,qabs2
    
    def cylinder_perpendicular_efficiencies(self,h):
        # height of cylinder: h
        # qsca2 = 2/self.x*(np.sum(np.abs(self.an2)**2+np.abs(self.bn2)**2)-self.an2[0]**2)
        # qext2 = np.real(2/self.x*np.sum(np.abs(self.bn2)**2))
        # qabs2=qext2-qsca2

        qsca2,qext2,qabs2 = self.cylinder_perpendicular_coefficient(h)
        # c = np.pi*self.r**2
        self.miu_s2=qsca2*self.vf*h*10000
        self.miu_a2=qabs2*self.vf*h*10000

        return self.miu_s2,self.miu_a2
    
    def cylinder_miut(self,d,ca,n_zeta,n_phi,n,stokes_in,qsca2,an1,an2,bn1,bn2):
        # d : diameter
        # CA : density
        numerator = 0
        denominator = 0
        new_q_sca = np.zeros((n_zeta,1))
        for i in range(1,n_zeta):
            zeta = i*np.pi/n_zeta
            numerator = 0
            denominator = 0
            for j in range(1,n_phi):
                phi = i*2*np.pi/n_phi
                T1,T2,T3,T4 = self.cylinder_cal_T(phi,an1,an2,bn1,bn2,n)
                M=2/(np.pi*self.x*np.sin(zeta))*self.cylinder_Smatrix(T1[i],T2[i],T3[i],T4[i])
                V1 = M*stokes_in
                numerator = numerator + V1[0]
                V2 = M*np.array([1,1,0,0])
                denominator = denominator + V2[0]
        
            new_q_sca[i] = d*ca*qsca2*numerator/denominator*10000

        return new_q_sca
    
    def cylinder_phase_function(self,h,nzeta,nphi):
        x = np.linspace(1,np.pi,nzeta)
        y = np.linspace(1,2*np.pi,nphi)

        # x = np.linspace(0,np.pi,nzeta)
        # y = np.linspace(0,2*np.pi,nphi)
        # TODO: dimension number setting!!
        theta,phi= np.meshgrid(x,y)
        # value 0 is dangerous!!!!

        S_term1 = (1+np.cos(theta))/np.pi
        c_div = mymath()
        S_term2 = c_div.complex_div(self.x*np.sin(self.x*np.sin(theta)*np.sin(phi)),(self.x*np.sin(theta)*np.sin(phi)))
        # (self.x*np.sin(self.x*np.sin(theta)*np.sin(phi)))/(self.x*np.sin(theta)*np.sin(phi))
        R = h/(2*self.r)
        S_term3 = c_div.complex_div((self.x*R*np.sin(self.x*R*np.sin(theta)*np.cos(phi))),(self.x*R*np.sin(theta)*np.cos(phi)))
        # (self.x*R*np.sin(self.x*R*np.sin(theta)*np.cos(phi)))/(self.x*R*np.sin(theta)*np.cos(phi))

        S = S_term1*S_term2*S_term3
        p = S**2/(4*self.x**2*R)

        return p

    # # def cylinder_mueller(self,phi,n):
    #     Bigtheta = np.pi - phi
    #     T1 = np.sum(2*self.bn1*np.cos(n*Bigtheta))-self.bn1[0]
    #     T2 = np.sum(2*self.an2*np.cos(n*Bigtheta))-self.an1[0]
    #     T3 = np.sum(-2j*self.an1*np.sin(n*Bigtheta))
    #     T4 = np.sum(-2j*self.bn2*np.sin(n*Bigtheta))

    #     # T11=(np.abs(T1)**2+np.abs(T2)**2)/2
    #     # T12=(np.abs(T1)**2-np.abs(T2)**2)/2
    #     # T33=np.real(T1*np.conjugate(T2))
    #     # T34=np.imag(T1*np.conjugate(T2))

    #     m11= (np.abs(T1)**2+np.abs(T2)**2+np.abs(T3)**2+np.abs(T4)**2)/2
    #     m12= (np.abs(T1)**2-np.abs(T2)**2+np.abs(T3)**2-np.abs(T4)**2)/2
    #     m13=np.real(T1*np.conjugate(T4)+T2*np.conjugate(T3))
    #     m14=np.imag(T1*np.conjugate(T4)-T2*np.conjugate(T3))
    #     m21= (np.abs(T1)**2-np.abs(T2)**2-np.abs(T3)**2+np.abs(T4)**2)/2
    #     m22= (np.abs(T1)**2+np.abs(T2)**2-np.abs(T3)**2-np.abs(T4)**2)/2
    #     m23=np.real(T1*np.conjugate(T4)-T2*np.conjugate(T3))
    #     m24=np.imag(T1*np.conjugate(T4)+T2*np.conjugate(T3))

    #     m31=np.real(T1*np.conjugate(T3)+T2*np.conjugate(T4))
    #     m32=np.real(T1*np.conjugate(T3)-T2*np.conjugate(T4))
    #     m33=np.real(np.conjugate(T1)*T2+np.conjugate(T3)*T4)
    #     m34=np.imag(T1*np.conjugate(T2)+T3*np.conjugate(T4))

    #     m41=np.imag(np.conjugate(T1)*T3+T2*np.conjugate(T4))
    #     m42=np.imag(np.conjugate(T1)*T3-T2*np.conjugate(T4))
    #     m43=np.imag(np.conjugate(T1)*T2-np.conjugate(T3)*T4)
    #     m44=np.real(np.conjugate(T1)*T2-np.conjugate(T3)*T4)

    #     M = np.array([m11,m12,m13,m14], [m21,m22,m23,m24], [m31,m32,m33,m34], [m41,m42,m43,m44])
    #     return M
    
    def cylinder_cal_T(self,phi,an1,an2,bn1,bn2,n):
        # phi=[ii*2*math.pi/(self.nphi-1) for ii in range(0,self.nphi)]
        Bigtheta = np.pi - phi
        T1 = np.sum(2*bn1*np.cos(n*Bigtheta))-bn1[0]
        T2 = np.sum(2*an2*np.cos(n*Bigtheta))-an1[0]
        T3 = np.sum(-2j*an1*np.sin(n*Bigtheta))
        T4 = np.sum(-2j*bn2*np.sin(n*Bigtheta))
        return T1, T2, T3, T4

    def cylinder_Smatrix(self, T1,T2,T3,T4):
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
    

    def cylinder_mie_treams(self,k0s,refrative_index=16 + 0.5j,mmax = 4,radius=75,kzs=[0]):
        # default value
        # k0s = 2 * np.pi * np.linspace(1 / 6000, 1 / 300, 200)
        materials = [treams.Material(refrative_index), treams.Material()]
        # mmax = 4
        # radius = 75
        # kzs = [0]
        cylinders = [treams.TMatrixC.cylinder(kzs, mmax, k0, radius, materials) for k0 in k0s]
        xw_sca = np.array([tm.xw_sca_avg for tm in cylinders]) / (2 * radius)
        xw_ext = np.array([tm.xw_ext_avg for tm in cylinders]) / (2 * radius)
        cwb_mmax0 = treams.CylindricalWaveBasis.default(kzs, 0)
        cylinders_mmax0 = [tm[cwb_mmax0] for tm in cylinders]
        xw_sca_mmax0 = np.array([tm.xw_sca_avg for tm in cylinders_mmax0]) / (2 * radius)
        xw_ext_mmax0 = np.array([tm.xw_ext_avg for tm in cylinders_mmax0]) / (2 * radius)
        return xw_sca,xw_ext
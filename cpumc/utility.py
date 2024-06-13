import math
from PIL import Image
# import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

class Nstop:

    def __init__(self) -> None:
        pass
    
    def cal_m(self,Npar,Nmed):
        m = Npar/Nmed
        return m
    
    def cal_propagator(self,n_medium,wavelength):
        k = 2 * math.pi*n_medium/wavelength
        return k
     
    def cal_nmax(self,Npar,Nmed,r,wavelength):
        m = Npar/Nmed
        k = self.cal_propagator(Nmed,wavelength)
        x = k * r
        # n-stop 
        nmax=round(2+x+4*math.pow(x,1/3))
        return nmax
    
        # v2
        # m = Npar/Nmed
        # k = 2 * math.pi*Nmed/Lambda 
        # x = k * r
        # # n-stop 
        # nmax=round(2+x+4*math.pow(x,1/3))

class mymath:
    def __init__(self) -> None:
        pass
    def is_square(self,n):
        square_root = math.sqrt(n)
        if int(square_root)**2 == n:
            return True
        else:
            return False
        
    def tozero(self,N):
        # for n in N:
        for idx, nx in enumerate(N):
            for idy, ny in enumerate(nx):
                if int(ny) != 0:
                    N[idx,idy] = 0
                else:
                    N[idx,idy] = ny
        return N
    
    def sixstokes2mueller(self,S,sizex,sizey,polarized_light_type,name):
        # i,j,4 
        S = np.real(S)
        types: ['LHP','LVP','Lp45','Lm45','RCP','LCP'] 
        for id, mode in enumerate(polarized_light_type):
            if mode=='LHP':
                LHP = S[id,:,:,:]
            elif mode=='LVP':
                LVP = S[id,:,:,:]
            elif mode=='Lp45': 
                LP45 = S[id,:,:,:]
            elif mode=='Lm45':
                LM45 = S[id,:,:,:]
            elif mode=='RCP': 
                RCP = S[id,:,:,:]
            elif mode=='LCP': 
                LCP = S[id,:,:,:]   

            # LP45 = S['LP45']
            # LM45 = S['LM45']
            # LCP = S['LCP']
            # RCP = S['RCP']


        # fig = plt.figure()
        # plt.subplot(3,4,1),plt.imshow(LHP[0,:,:] ,cmap='jet'),plt.title('LHP0')
        # plt.subplot(3,4,2),plt.imshow(LHP[1,:,:] ,cmap='jet'),plt.title('LHP1')
        # plt.subplot(3,4,3),plt.imshow(LHP[2,:,:] ,cmap='jet'),plt.title('LHP2')
        # plt.subplot(3,4,4),plt.imshow(LHP[3,:,:] ,cmap='jet'),plt.title('LHP3')
        
        # plt.subplot(3,4,5),plt.imshow(LVP[0,:,:] ,cmap='jet'),plt.title('LVP0')
        # plt.subplot(3,4,6),plt.imshow(LVP[1,:,:] ,cmap='jet'),plt.title('LVP1')
        # plt.subplot(3,4,7),plt.imshow(LVP[2,:,:] ,cmap='jet'),plt.title('LVP2')
        # plt.subplot(3,4,8),plt.imshow(LVP[3,:,:] ,cmap='jet'),plt.title('LVP3')

        # plt.subplot(3,4,9),plt.imshow(LP45[0,:,:] ,cmap='jet'),plt.title('LP450')
        # plt.subplot(3,4,10),plt.imshow(LP45[1,:,:] ,cmap='jet'),plt.title('LP451')
        # plt.subplot(3,4,11),plt.imshow(LP45[2,:,:] ,cmap='jet'),plt.title('LP452')
        # plt.subplot(3,4,12),plt.imshow(LP45[3,:,:] ,cmap='jet'),plt.title('LP453')
        # plt.show()


        # fig = plt.figure()
        # plt.subplot(3,4,1),plt.imshow(LM45[0,:,:] ,cmap='jet'),plt.title('LP450')
        # plt.subplot(3,4,2),plt.imshow(LM45[1,:,:] ,cmap='jet'),plt.title('LP451')
        # plt.subplot(3,4,3),plt.imshow(LM45[2,:,:] ,cmap='jet'),plt.title('LP452')
        # plt.subplot(3,4,4),plt.imshow(LM45[3,:,:] ,cmap='jet'),plt.title('LP453')

        # plt.subplot(3,4,5),plt.imshow(LCP[0,:,:] ,cmap='jet'),plt.title('LCP0')
        # plt.subplot(3,4,6),plt.imshow(LCP[1,:,:] ,cmap='jet'),plt.title('LCP1')
        # plt.subplot(3,4,7),plt.imshow(LCP[2,:,:] ,cmap='jet'),plt.title('LCP2')
        # plt.subplot(3,4,8),plt.imshow(LCP[3,:,:] ,cmap='jet'),plt.title('LCP3')
        
        # plt.subplot(3,4,9),plt.imshow(RCP[0,:,:] ,cmap='jet'),plt.title('RCP0')
        # plt.subplot(3,4,10),plt.imshow(RCP[1,:,:] ,cmap='jet'),plt.title('RCP1')
        # plt.subplot(3,4,11),plt.imshow(RCP[2,:,:] ,cmap='jet'),plt.title('RCP2')
        # plt.subplot(3,4,12),plt.imshow(RCP[3,:,:] ,cmap='jet'),plt.title('RCP3')
        # plt.show()
        
        M = np.zeros((sizex,sizey,4,4))
        # m00
        M[:,:,0,0] = (LHP[0,:,:]+LVP[0,:,:])/2
        # m01 
        M[:,:,0,1] = (LHP[0,:,:]-LVP[0,:,:])/2
        # m10
        M[:,:,1,0] = (LHP[1,:,:]+LVP[1,:,:])/2
        # m11
        M[:,:,1,1] = (LHP[1,:,:]-LVP[1,:,:])/2
        # m20 
        M[:,:,2,0] = (LHP[2,:,:]+LVP[2,:,:])/2
        # m21 
        M[:,:,2,1] = (LHP[2,:,:]-LVP[2,:,:])/2
        # m30 
        M[:,:,3,0] = (LHP[3,:,:]+LVP[3,:,:])/2
        # m31 
        M[:,:,3,1] = (LHP[3,:,:]-LVP[3,:,:])/2

        # m00 = (LP45[:,:,0]+LM45[:,:,0])/2
        # m02 
        M[:,:,0,2] = (LP45[0,:,:]-LM45[0,:,:])/2
        # m10 = (LP45[:,:,1]+LM45[:,:,1])/2
        # m12 
        M[:,:,1,2] = (LP45[1,:,:]-LM45[1,:,:])/2
        # m20 = (LP45[:,:,2]+LM45[:,:,2])/2
        # m22 
        M[:,:,2,2] = (LP45[2,:,:]-LM45[2,:,:])/2
        # m30 = (LP45[:,:,3]+LM45[:,:,3])/2
        # m32 
        M[:,:,3,2] = (LP45[3,:,:]-LM45[3,:,:])/2

        
        # m00 = (LCP[:,:,0]+RCP[:,:,0])/2
        # m03 
        M[:,:,0,3]= -(LCP[0,:,:]-RCP[0,:,:])/2
        #m10 = (LCP[:,:,1]+RCP[:,:,1])/2
        # m13 
        M[:,:,1,3]= -(LCP[1,:,:]-RCP[1,:,:])/2
        # m20 = (LCP[:,:,2]+RCP[:,:,2])/2
        # m23 
        M[:,:,2,3]= -(LCP[2,:,:]-RCP[2,:,:])/2
        # m30 = (LCP[:,:,3]+RCP[:,:,3])/2
        # m33 
        M[:,:,3,3]= -(LCP[3,:,:]-RCP[3,:,:])/2
        



        # norm=M[:,:,0,0]
        # mask = ma.make_mask(3*norm)
        # # MM[:,:,row,colum,idW] = np.clip(MM[:,:,row,colum,idW],-1,1)
        # # fig = plt.figure()
        # for i in range(100):
        #     for j in range(100):
        #         for row in range(0,4):
        #             for colum in range(0,4):
        #                 if (row==0)&(colum==0):
        #                     if mask[i,j]:
        #                         M[i,j,row,colum] = M[i,j,row,colum]/M[i,j,0,0]
        #                     else:
        #                         M[i,j,row,colum]= 0
        #                 else:
        #                     if mask[i,j]:
        #                         M[i,j,row,colum] = 1
        #                     else:
        #                         M[i,j,row,colum] = 0


        # MM= np.clip(M,-1,1)

        
        # div = M[:,:,0,0]
        
        # for row in range(0,4):
        #     for colum in range(0,4):                 
        #         M[:,:,row,colum] = M[:,:,row,colum]/div
        
        # MM= np.clip(M,-1,1)
        
        M = np.real(M)
        fig = plt.figure()
        plt.subplot(4,4,1),plt.imshow(M[:,:,0,0] ,cmap='jet'),plt.title('m00')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,2),plt.imshow(M[:,:,0,1] ,cmap='jet'),plt.title('m01')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,3),plt.imshow(M[:,:,0,2] ,cmap='jet'),plt.title('m02')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,4),plt.imshow(M[:,:,0,3] ,cmap='jet'),plt.title('m03')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,5),plt.imshow(M[:,:,1,0] ,cmap='jet'),plt.title('m10')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,6),plt.imshow(M[:,:,1,1] ,cmap='jet'),plt.title('m11')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,7),plt.imshow(M[:,:,1,2] ,cmap='jet'),plt.title('m12')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,8),plt.imshow(M[:,:,1,3] ,cmap='jet'),plt.title('m13')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,9),plt.imshow(M[:,:,2,0] ,cmap='jet'),plt.title('m20')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,10),plt.imshow(M[:,:,2,1] ,cmap='jet'),plt.title('m21')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,11),plt.imshow(M[:,:,2,2] ,cmap='jet'),plt.title('m22')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,12),plt.imshow(M[:,:,2,3] ,cmap='jet'),plt.title('m23')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,13),plt.imshow(M[:,:,3,0] ,cmap='jet'),plt.title('m30')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,14),plt.imshow(M[:,:,3,1] ,cmap='jet'),plt.title('m31')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,15),plt.imshow(M[:,:,3,2] ,cmap='jet'),plt.title('m32')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        plt.subplot(4,4,16),plt.imshow(M[:,:,3,3] ,cmap='jet'),plt.title('m33')
        colorbar = plt.colorbar()
        # plt.clim(-1,1)
        # plt.show()

        fig.savefig('MonteCarlo/cpumc/result/v9/figure_ori'+str(name)+'.png')
        np.save('MonteCarlo/cpumc/result/v9/ori_'+str(name), M)

        
        # norm -1~1
        M1 = M
        norm=M1[:,:,0,0]
        fig = plt.figure()
        for row in range(0,4):
            for colum in range(0,4):
        #         # norm=np.amax(MM[:,:,row,colum,idW])
                M1[:,:,row,colum] = M1[:,:,row,colum]/norm
                # M1[:,:,row,colum] = np.clip(M1[:,:,row,colum],-1,1)
                plt.subplot(4,4,row*4+(colum+1)),plt.imshow(M1[:,:,row,colum] ,cmap='jet'),plt.title('M'+str(row)+str(colum))
                colorbar = plt.colorbar()
                plt.clim(-1,1)
                plt.subplot(4,4,1),plt.imshow(M1[:,:,0,0],cmap='jet'),plt.title('M00')
                colorbar = plt.colorbar()
                plt.clim(-1,1)
                plt.subplot(4,4,16),plt.imshow(M1[:,:,3,3],cmap='jet'),plt.title('M33')
                colorbar = plt.colorbar()
                plt.clim(-1,1)
        fig.savefig('MonteCarlo/cpumc/result/v9/figure_nor1'+str(name)+'.png')
        np.save('MonteCarlo/cpumc/result/v9/nor1_'+str(name), M1)

        # norm -0.1~0.1
        M2 = M
        norm=M2[:,:,0,0]
        fig = plt.figure()
        for row in range(0,4):
            for colum in range(0,4):
        #         # norm=np.amax(MM[:,:,row,colum,idW])
                M2[:,:,row,colum] = M2[:,:,row,colum]/norm
                # M2[:,:,row,colum] = np.clip(M2[:,:,row,colum],-0.1,0.1)
                plt.subplot(4,4,row*4+(colum+1)),plt.imshow(M2[:,:,row,colum] ,cmap='jet'),plt.title('M'+str(row)+str(colum))
                colorbar = plt.colorbar()
                plt.clim(-0.1,0.1)
                plt.subplot(4,4,1),plt.imshow(M2[:,:,0,0],cmap='jet'),plt.title('M00')
                colorbar = plt.colorbar()
                plt.clim(-0.1,0.1)
                plt.subplot(4,4,16),plt.imshow(M2[:,:,3,3],cmap='jet'),plt.title('M33')
                colorbar = plt.colorbar()
                plt.clim(-0.1,0.1)

        fig.savefig('MonteCarlo/cpumc/result/v9/figure_nor2'+str(name)+'.png')
        np.save('MonteCarlo/cpumc/result/v9/nor2_'+str(name), M2)
        
        return M
    
    def complex_div(self,x,y):
        a = np.real(x)
        b = np.imag(x)
        c = np.real(y)
        d = np.imag(y)
        output = (a*c+b*d)+1j*(b*c-a*d)/(c**2+d**2)
        return output

    def complex_multi(self,x,y):
        a = np.real(x)
        b = np.imag(x)
        c = np.real(y)
        d = np.imag(y)
        output = (a*c-b*d)+1j*(b*c+a*d)
        return output
    
    def cal_angle_differece(self,a,b):
        b=np.array(b)
        dot_product =np.dot(a.ravel(),b.ravel())
        mag_a = np.linalg.norm(a.ravel())
        mag_b = np.linalg.norm(b.ravel())

        angle = np.arccos(dot_product/(mag_a*mag_b))
        return angle
    
class measurement():
    def __init__(self) -> None:
        pass
    def record_depth(self,p_max,p_actual):
        if p_actual>p_max:
            return p_actual
        else:
            return p_max

class mysave:
    def __init__(self) -> None:
        pass
    def arr2im(self,data, fname):
        out = Image.new('RGB', data.shape[1::-1])
        out.putdata(map(tuple, data.reshape(-1, 3)))
        out.save(fname)

class image_process():
    def __init__(self) -> None:
        pass
    def adjust_intensity(image,Imax,Imin):
        # normalized to [0,1] first
        normalized_image = image.astype(np.float32)/255.0
        pass 
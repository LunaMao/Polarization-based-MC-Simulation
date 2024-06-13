# User instruction

# 1. Author: Luna (Yuxuan Mao)

# 2. Date: from 11/15 to ...

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

# import cv2
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt


class imaging_enhancement():
        def __init__(self):
            self.basic_coeff=0.16667

        def basic_enhance(self,MM,stokes_camera,col1=[0,1,2,3],col2=[0,0,0,0],coeff=[0.3,0.3,0.3,0.3]):
                        #   coeff=[0.16667,0.16667,0.16667,0.16667]):
            # col1=[0,1,2,3]
            # col2=[0,0,0,0] 
            MM[:,:,col1[0],col2[0]]=MM[:,:,0,col2[0]]+stokes_camera[0,:,:]*coeff[0]
            MM[:,:,col1[1],col2[1]]=MM[:,:,1,col2[0]]+stokes_camera[1,:,:]*coeff[1]
            MM[:,:,col1[2],col2[2]]=MM[:,:,2,col2[0]]+stokes_camera[2,:,:]*coeff[2]
            MM[:,:,col1[3],col2[3]]=MM[:,:,3,col2[0]]+stokes_camera[3,:,:]*coeff[3]
            return MM
        
        def polarchannel_adjust(self,MM,stokes_camera,mode):
            if mode == 'LHP':
                second_coeff=[0.5,0.5,0.5,0.5]
                col1=[0,1,2,3]
                col2=[1,1,1,1]
            elif mode == 'LVP':
                second_coeff=[-0.5,-0.5,-0.5,-0.5]
                col1=[0,1,2,3]
                col2=[1,1,1,1]
            elif mode == 'Lp45':
                second_coeff=[0.5,0.5,0.5,0.5]
                col1=[0,1,2,3]
                col2=[2,2,2,2]
            elif mode == 'Lm45':
                second_coeff=[-0.5,-0.5,-0.5,-0.5]
                col1=[0,1,2,3]
                col2=[2,2,2,2]
            elif mode == 'RCP':
                second_coeff=[0.5,0.5,0.5,0.5]
                col1=[0,1,2,3]
                col2=[3,3,3,3]
            elif mode == 'LCP':
                second_coeff=[-0.5,-0.5,-0.5,-0.5]
                col1=[0,1,2,3]
                col2=[3,3,3,3]
            
        # def polar_channel_adjust(self,MM,stokes_camera,mode):
        #     for channel in range(0,6):
        #         if channel == 0:
        #             second_coeff=[0.5,0.5,0.5,0.5]
        #             col1=[0,1,2,3]
        #             col2=[1,1,1,1]
        #             MM= self.basic_enhance(MM,stokes_camera,col1,col2,second_coeff)
        #         if channel == 1:
        #             second_coeff=[0.5,0.5,0.5,0.5]
        #             col1=[0,1,2,3]
        #             col2=[1,1,1,1]
        #             MM= self.basic_enhance(MM,stokes_camera,col1,col2,second_coeff)





        #     elif mode == 'LVP':
        #         second_coeff=[-0.5,-0.5,-0.5,-0.5]
        #         col1=[0,1,2,3]
        #         col2=[1,1,1,1]
        #     elif mode == 'Lp45':
        #         second_coeff=[0.5,0.5,0.5,0.5]
        #         col1=[0,1,2,3]
        #         col2=[2,2,2,2]
        #     elif mode == 'Lm45':
        #         second_coeff=[-0.5,-0.5,-0.5,-0.5]
        #         col1=[0,1,2,3]
        #         col2=[2,2,2,2]
        #     elif mode == 'RCP':
        #         second_coeff=[0.5,0.5,0.5,0.5]
        #         col1=[0,1,2,3]
        #         col2=[3,3,3,3]
        #     elif mode == 'LCP':
        #         second_coeff=[-0.5,-0.5,-0.5,-0.5]
        #         col1=[0,1,2,3]
        #         col2=[3,3,3,3]
        #     return self.basic_enhance(MM,stokes_camera,col1,col2,second_coeff)
        
class frequency_analysis():
        def __init__(self,image):
            self.image=image
        def frequency_spectrum(self):
            f = np.fft.fft2(self.image)
            fshift = np.fft.fftshift(f)
            # absoulute value: real part 
            # amplitute
            amp = np.log(np.abs(f))
            # amplitute but move to the centre
            amp_centre = np.log(np.abs(fshift))
            # phase
            phase = np.angle(f)
            phase_centre = np.angle(fshift)

            fig = plt.figure()
            plt.subplot(2,2,1),plt.imshow(self.image,cmap='jet'),plt.title('original')
            plt.subplot(2,2,2),plt.imshow(amp,cmap='jet'),plt.title('amplitude')
            plt.subplot(2,2,3),plt.imshow(amp_centre,cmap='jet'),plt.title('centre')
            plt.subplot(2,2,4),plt.imshow(phase_centre,cmap='jet'),plt.title('phase')
            plt.show()
            fig.savefig('frequency_analysis.png')

class psfanaysis():
    def __init__(self) -> None:
         pass 
    def dipole_orientation(self):
        pass
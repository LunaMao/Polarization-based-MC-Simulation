# Polarized _monte_carlo_Meridian Planes
# Author: Yuxuan Mao, Albert Van Eeckhout Alsinet
# Start Date: 2023-11-7
# Copyright: 
# Optical Metrology & Image Processing Laboratory (MIP-OptiLab)
# Supported by the China Scholarship Council

import yaml
import sys
sys.path.append('mie.py')
from mie import Mie_scattering
from Probabilisty import Sphere_probabilisty,cylinder_prosibility
from light import polarized_light,polarized_mueller_matrix
from transmission_torch import transmission
import math 
import time
import multiprocessing
from operator import methodcaller
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera import imaging_enhancement,frequency_analysis
from utility import Nstop, mysave,mymath
# import treams
import scipy.fftpack as fp
import numpy as np
import depolarizing
from alive_progress import alive_bar
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# set TORCH_USE_CUDA_DSA

def main():

        # Check Computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype = torch.cuda.FloatTensor
        print("using GPU:",torch.cuda.get_device_name(0))
    else:
        device =torch.device('cpu')
        dtype = torch.FloatTensor
        print("Using CPU")

    # step 0 get the start time
    st = time.time()

    #  Step 1 Initialize the parameters
    with open('MonteCarlo/torchmc/config.yml', 'r') as file:
        parameters = yaml.safe_load(file)

    # constant of possibility
    nalfa = parameters['Probabilisty']['nalfa']
    nbeta = parameters['Probabilisty']['nbeta']

    # angle distribution
    nzeta = parameters['Probabilisty']['cylinder']['nzeta']
    nphi = parameters['Probabilisty']['cylinder']['ntheta']

    # th for absorption    
    th = parameters['Initialized_parameters']['th']

    # space limit of sandbox
    d =parameters['Initialized_parameters']['d']
    x_max =parameters['Initialized_parameters']['xmax']
    y_max =parameters['Initialized_parameters']['ymax']

    # light source prepare
    
    # photon 
    N_photons=parameters['photon']['number']
    pattern =parameters['light_source']['pattern']
    beam_radius =parameters['light_source']['beam_radius']

    P_default =parameters['photon']['p']
    D_default =parameters['photon']['d']

    p_vectors = torch.zeros(N_photons,3).type(dtype)
    p_vectors[:,0]=P_default[0]
    p_vectors[:,1]=P_default[1]
    p_vectors[:,2]=P_default[2]

    d_vectors = torch.zeros(N_photons,3).type(dtype)
    d_vectors[:,0]=D_default[0]
    d_vectors[:,1]=D_default[1]
    d_vectors[:,2]=D_default[2]

    I = parameters['light_source']['I']
    
    # custom light property
    e0x = parameters['light_source']['e0x']
    e0y = parameters['light_source']['e0y']
    retardance = parameters['light_source']['retardance']

    traverse = parameters['light_source']['if_traverse']
    if traverse:
        polarized_light_type = parameters['light_source']['types']
    else:
        polarized_light_type = parameters['light_source']['type']

    spectrum = parameters['light_source']['Lambda']
    incident_light = polarized_mueller_matrix(I,e0x,e0y,retardance)
    
    # Camera prepare
    resolution_x = parameters['camera']['resolution_x']
    resolution_y = parameters['camera']['resolution_y']
    MM = torch.zeros(resolution_x,resolution_y,4,4,len(spectrum)).type(dtype)
    MM1 = torch.zeros(resolution_x,resolution_y,4,4,len(spectrum)).type(dtype)
    D = torch.zeros(len(spectrum),2).type(dtype)
    # Possible photon events
    type_list = parameters['scattering']['type_list']

    batch_size =parameters['Aceelerate']['batch_size']

    for idW,Lambda in enumerate(spectrum):
        # print("incident light information: intensity {}, polarization states {}, wavelength {}nm".format(I,mode,int(Lambda*1e3)))
        # each polarized states
        bs_camera = torch.zeros(6, resolution_x,resolution_y, 4).type(dtype)
        t_camera = torch.zeros(6, resolution_x,resolution_y,4).type(dtype)
        # np.zeros((6, 4, resolution_x,resolution_y),dtype=np.complex_)
        # t_camera = np.zeros((6, 4, resolution_x,resolution_y),dtype=np.complex_)
        with alive_bar(len(polarized_light_type)) as bar:
            for idm, mode in enumerate(polarized_light_type):
                # # each polarized states
                # for idm, mode in enumerate(polarized_light_type):

                # #  multi-spectral imaging
                # for idW,Lambda in  enumerate(spectrum):
                print("incident light information: intensity {}, polarization states {}, wavelength {}nm".format(I,mode,int(Lambda*1e3)))
                stokes= incident_light.gpl(mode)
                #  pre-calculate for each photon events
                for idx, event in enumerate(type_list):
                    Npar = parameters['scattering'][event]['Npar']
                    Nmed = parameters['scattering'][event]['Nmed']
                    r = parameters['scattering'][event]['r']
                    vf = parameters['scattering'][event]['vf']
                    #  maximum iteration times 
                    Tool=Nstop()
                    nmax = Tool.cal_nmax(Npar,Nmed,r,Lambda)
                    # for different shapes
                    if event.startswith('sphere'):
                        if parameters['scattering'][event]['method'] == 'traditional':
                            # mie scattering
                            Mie=Mie_scattering(Tool.cal_m(Npar,Nmed),Tool.cal_propagator(Nmed,Lambda),vf,r)
                            n,an,bn=Mie.Sphere_mie_an_bn()
                            # print("Sphere scattering: order {}, an {}, bn {}".format(n,an,bn))
                            miu_s,miu_a=Mie.Sphere_coefficient()
                            # print("miu_scattering {}, miu_absorption {}, ".format(miu_s,miu_a))

                            # define the probability based on scattered-stokes 
                            Probabilisty=Sphere_probabilisty(nalfa,nbeta,n,an,bn,nmax)
                            Scattering_Matrix=Probabilisty.Scattering_matrix()
                            # print(Scattering_Matrix)
                            Palf_norm=Probabilisty.cal_norm_P_alfa(Scattering_Matrix)
                            # notes:
                            # the off-angle changes with the stokes_in, so we can't calculate in the beginning stage
                            # alfa,beta,stokes = Probabilisty.cal_sca_angle(Scattering_Matrix,stokes)
                    
                            Scattering=transmission(miu_s,miu_a,d,th,x_max,y_max,N_photons,Scattering_Matrix,stokes,Palf_norm,nalfa,nbeta,p_vectors,d_vectors,pattern,beam_radius,batch_size)
                            # MC simulation and imaging
                            t_camera[idm,:,:,:],bs_camera[idm,:,:,:], invaild_number = Scattering.Trans(resolution_x,resolution_y)
                            # print(stokes_camera)
                
                            # 4 measurement
                            # D[idW,0] = depth_max
                            # D[idW,1] = effective_number

                        elif parameters['scattering'][event]['method'] == 'T_matrix':
                        #     k0s=[2*np.pi/(s*1000) for s in spectrum]
                        #     # k0s = 2 * np.pi * np.linspace(1 / 700, 1 / 300, 200)
                        #     Mie=Mie_scattering(Tool.cal_m(Npar,Nmed),Tool.cal_propagator(Nmed,Lambda),vf,r)
                        #     n,an,bn=Mie.Sphere_mie_an_bn()
                        #     qsca,qext = Mie.sphere_mie_treams(k0s, Npar, 4, r)
                        #     qabs=qext-qsca
                        #     miu_s = 3*qsca[idW]*vf*1e4/(r*4)
                        #     miu_a = 3*qabs[idW]*vf*1e4/(r*4)
                        #     Probabilisty=Sphere_probabilisty(nalfa,nbeta,n,an,bn,nmax)
                        #     Scattering_Matrix=Probabilisty.Scattering_matrix()
                        #     # print(Scattering_Matrix)
                        #     Palf_norm=Probabilisty.cal_norm_P_alfa(Scattering_Matrix)
                        #     pass
                        # else:
                            raise Exception("Unspported method!")
                    elif event.startswith('cylinder'):
                        if parameters['scattering'][event]['method'] == 'traditional':
                            
                            h = parameters['scattering'][event]['length']
                            l = parameters['scattering'][event]['main-axix-direction']
                            nzeta = parameters['Probabilisty']['cylinder']['nzeta']
                            nphi = parameters['Probabilisty']['cylinder']['ntheta']
                            k = Tool.cal_propagator(Nmed,Lambda)

                            Mie=Mie_scattering(Tool.cal_m(Npar,Nmed),k,vf,r)
                            x = k*r
                            nmax=round(2+x+4*math.pow(x,1/3))
                            # n-stop use functions in utility
                            an1=np.zeros((nzeta,nmax),dtype='complex_')
                            an2=np.zeros((nzeta,nmax),dtype='complex_')
                            bn1=np.zeros((nzeta,nmax),dtype='complex_')
                            bn2=np.zeros((nzeta,nmax),dtype='complex_')
                            Zeta=[(ii)*math.pi/(nzeta) for ii in range(1,nzeta)]
                            # result1 = [Mie.cylinder_mie_an_bn_c1(zeta) for zeta in Zeta]
                            for id_zeta, zeta in enumerate(Zeta):
                                nmax, an1[id_zeta], bn1[id_zeta]= Mie.cylinder_mie_an_bn_c1(zeta)
                                # print("Cylinder scattering: order {}, an1 {}, bn1 {}".format(n,an1,bn1))
                                # TODO: nan ?? why??
                                nmax, an2[id_zeta], bn2[id_zeta]= Mie.cylinder_mie_an_bn_c2(zeta)
                                # print("Cylinder scattering: order {}, an2 {}, bn2 {}".format(n,an2,bn2))
                            
                            miu_s1,miu_a1= Mie.cylinder_parallel_efficiencies(h)
                            miu_s2,miu_a2= Mie.cylinder_perpendicular_efficiencies(h)

                            miu_s = [miu_s1, miu_s2]
                            miu_a = [miu_a1, miu_a2]
                            # print("miu_scattering1 {}, miu_absorption1 {}, ".format(miu_s1,miu_a1))
                            # print("miu_scattering2 {}, miu_absorption2 {}, ".format(miu_s2,miu_a2))
                            #  
                            P = np.real(Mie.cylinder_phase_function(h,nzeta,nphi))

                            # define the probability based on scattered-stokes 
                            Probabilisty=cylinder_prosibility(an1,an2,bn1,bn2,nzeta,nphi)

                            P_zeta_norm = Probabilisty.cal_norm_P_zeta(P)
                            Scattering_Matrix= np.zeros((4,360),dtype=np.complex_)

                            cylinder_scattering=transmission(miu_s,miu_a,d,th,x_max,y_max,N_photons,Scattering_Matrix,stokes,P_zeta_norm,nzeta,nphi,p_vectors,d_vectors,pattern,beam_radius)
                            qsca,_,_=Mie.cylinder_parallel_coefficient(h)
                            print("finised pre-calculation!")
                            # miu_t_initalized = Mie.cylinder_miut(2*r,vf,nzeta,nphi,stokes,qsca2,an1,an2,bn1,bn2)
                            # miu_t for every angle...but for initailized polarized state
                            # k=2*np.pi/Lambda
                            t_camera[idm,:,:,:],bs_camera[idm,:,:,:] = cylinder_scattering.Trans_test(resolution_x,resolution_y,P_zeta_norm, nzeta, nphi,p_vectors, d_vectors,r,2*np.pi/Lambda,an1,an2,bn1,bn2,stokes,qsca,l,vf,P)
                            # resolution_x, resolution_y, Palf_norm, nzeta, nphi, r, k, an1, an2, bn1, bn2, stokes_in, qsca, l, vf, P
                            # t_camera[idm,:,:,:],bs_camera[idm,:,:,:], depth_max, effective_number = cylinder_scattering.Partrans_cylinder(resolution_x,resolution_y,P_zeta_norm, nzeta, nphi,r,vf,2*np.pi/Lambda,an1,an2,bn1,bn2,stokes,qsca,l,P)
                            # T1,T2,T3,T4 =Probabilisty.cylinder_cal_T()
                            # print('t_camera[idm,:,:,:]')
                            # print(T1)
                            # Scattering_Matrix=Probabilisty.Scattering_matrix()
                            # print(Scattering_Matrix)
                            # Palf_norm=Probabilisty.cal_norm_P_alfa(Scattering_Matrix)
                        
                        elif parameters['scattering'][event]['method'] == 'T_matrix':
                            raise Exception("Unspported method!")
                    else:
                        raise Exception("Unspported shape!")
                bar()
            # original code
            # Scattering=transmission(miu_s,miu_a,d,th,x_max,y_max,N_photons,Scattering_Matrix,stokes,Palf_norm,nalfa,nbeta,p_vectors,d_vectors,pattern,beam_radius)
            # MC simulation and imaging
            # t_camera[idm,:,:,:],bs_camera[idm,:,:,:], depth_max, effective_number = Scattering.ParTrans(resolution_x,resolution_y)

            
            # 4 measurement
            # D[idW,0] = depth_max
            # D[idW,1] = effective_number
        
        # get the end time
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        # print('elapsed_time')


        # 5 plot the result
        s2m = mymath() 
        name = 'bs_'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'_'+str(elapsed_time)
        MM[:,:,:,:,idW]=s2m.sixstokes2mueller(bs_camera,resolution_x,resolution_y,polarized_light_type,name)
        
        name1 = 't_'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'_'+str(elapsed_time)
        
        MM1[:,:,:,:,idW]=s2m.sixstokes2mueller(t_camera,resolution_x,resolution_y,polarized_light_type,name1)

        # enchance = imaging_enhancement()
        # MM[:,:,:,:,idW]=np.real(MM[:,:,:,:,idW])
        # MM1[:,:,:,:,idW]=np.real(MM1[:,:,:,:,idW])
        # MM[:,:,:,:,idW]=enchance.basic_enhance(MM[:,:,:,:,idW],bs_camera)
        # MM[:,:,:,:,idW]=enchance.polarchannel_adjust(MM[:,:,:,:,idW],bs_camera,mode)
        # MM1[:,:,:,:,idW]=enchance.basic_enhance(MM1[:,:,:,:,idW],t_camera)
        # MM1[:,:,:,:,idW]=enchance.polarchannel_adjust(MM1[:,:,:,:,idW],t_camera,mode)
    



    # normal camera
    # for idW,Lambda in  enumerate(spectrum):

        # backscattering



        # # MM[:,:,:,:,idW]=np.real(MM[:,:,:,:,idW])
        # norm=MM[:,:,0,0,idW]
        # fig = plt.figure()
        # for row in range(0,4):
        #     for colum in range(0,4):
        # #         # norm=np.amax(MM[:,:,row,colum,idW])
        #         MM[:,:,row,colum,idW] = MM[:,:,row,colum,idW]/norm
        #         MM[:,:,row,colum,idW] = np.clip(MM[:,:,row,colum,idW],-0.1,0.1)
        #         plt.subplot(4,4,row*4+(colum+1)),plt.imshow(MM[:,:,row,colum,idW] ,cmap='jet'),plt.title('M'+str(row)+str(colum))
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        #         plt.subplot(4,4,1),plt.imshow(MM[:,:,0,0,idW],cmap='jet'),plt.title('M00')
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        #         plt.subplot(4,4,16),plt.imshow(MM[:,:,3,3,idW],cmap='jet'),plt.title('M33')
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        # # #  Save 
        
        # fig.savefig('result/v7/figure_bs'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'_'+str(D[idW,1])+'_'+str(D[idW,0])+'_'+str(elapsed_time)+'.png')
        # np.save('result/v7/bs'+str(N_photons)+'_'+str(Lambda)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(vf)+'_'+str(Npar)+'_'+str(event)+'_'+str(x_max)+'_'+str(D[idW,1]), MM[:,:,:,:,idW])
        
        
        # plt.show()

        # # trans
        # fig = plt.figure()
        # for row in range(0,4):
        #     for colum in range(0,4):
        #         MM1[:,:,row,colum,idW] = MM1[:,:,row,colum,idW]/norm
        #         MM1[:,:,row,colum,idW] = np.clip(MM1[:,:,row,colum,idW],-0.1,0.1)
        #         plt.subplot(4,4,row*4+(colum+1)),plt.imshow(MM1[:,:,row,colum,idW] ,cmap='jet'),plt.title('M'+str(row)+str(colum))
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        #         plt.subplot(4,4,1),plt.imshow(MM1[:,:,0,0,idW],cmap='jet'),plt.title('M00')
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        #         plt.subplot(4,4,16),plt.imshow(MM1[:,:,3,3,idW],cmap='jet'),plt.title('M33')
        #         colorbar = plt.colorbar()
        #         plt.clim(-0.1,0.1)
        # #  Save 
        # fig.savefig('result/v7/figure_Trans_'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'_'+str(D[idW,1])+'_'+str(D[idW,0])+'_'+str(elapsed_time)+'.png')
        # np.save('result/v7/Trans_'+str(N_photons)+'_'+str(Lambda)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(vf)+'_'+str(Npar)+'_'+str(event)+'_'+str(x_max)+'_'+str(D[idW,1]), MM1[:,:,:,:,idW])


        # frequency- camera
        ## Functions to go from image to frequency-image and back
        # 
        # frequency_camera = frequency_analysis(MM[:,:,0,0])
        # frequency_camera = frequency_analysis(MM[:,:,1,0])

        # im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
        #    axis=1)
        # freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
        #  axis=0)
        # frequency_camera.frequency_spectrum()


        # fig.savefig('result/v6/figure_'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'_'+str(D[idW,1])+'_'+str(D[idW,0])+'_'+str(elapsed_time)+'.png')




        # # depolarizing and related analysis
        # Mt = np.nan_to_num(MM[:,:,:,:,idW].reshape(resolution_x*resolution_y,4,4))
        
        # depolarizing_analysis = depolarizing.depolarizing(Mt,resolution_x*resolution_y)
        # P_delta_t = depolarizing_analysis.depolarization_index()
        # Pt = depolarizing_analysis.ipp()
        # P_vt,D_vt, Ps_t = depolarizing_analysis.cp()
        
        # P1 = Pt[:,0].reshape(resolution_x,resolution_y)
        # P2 = Pt[:,1].reshape(resolution_x,resolution_y)
        # P3 = Pt[:,2].reshape(resolution_x,resolution_y)
        # P_v = P_vt.reshape(resolution_x,resolution_y)
        # D_v = D_vt.reshape(resolution_x,resolution_y)
        # Ps = Ps_t.reshape(resolution_x,resolution_y)


        # fig = plt.figure()
        # plt.subplot(2,3,1),plt.imshow(P1 ,cmap='jet'),plt.title('P1')
        # colorbar = plt.colorbar()
        # plt.subplot(2,3,2),plt.imshow(P2 ,cmap='jet'),plt.title('P2')
        # colorbar = plt.colorbar()
        # plt.subplot(2,3,3),plt.imshow(P3 ,cmap='jet'),plt.title('P3')
        # colorbar = plt.colorbar()
        # plt.subplot(2,3,4),plt.imshow(P_v ,cmap='jet'),plt.title('P_v')
        # colorbar = plt.colorbar()
        # plt.subplot(2,3,5),plt.imshow(D_v ,cmap='jet'),plt.title('D_v')
        # colorbar = plt.colorbar()
        # plt.subplot(2,3,6),plt.imshow(Ps ,cmap='jet'),plt.title('Ps')
        # colorbar = plt.colorbar()
        # fig.savefig('result/v7/DP_'+str(Lambda)+'_'+str(N_photons)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(r)+'_'+str(vf)+'_'+str(x_max)+'.png')
        # np.save('result/v7/DP_'+str(N_photons)+'_'+str(Lambda)+'_'+str(pattern)+'_'+str(beam_radius)+'_'+str(vf)+'_'+str(Npar)+'_'+str(event)+'_'+str(x_max)+'_'+str(D[idW,1]), MM[:,:,:,:,idW])

if __name__ == "__main__":
    # Main program
    main()

    
    
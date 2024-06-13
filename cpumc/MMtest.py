# Polarized _monte_carlo_Meridian Planes
# Author: Yuxuan Mao, Albert Van Eeckhout Alsinet
# Start Date: 2023-11-7
# Copyright: 
# Optical Metrology & Image Processing Laboratory (MIP-OptiLab)
# Supported by the China Scholarship Council

import numpy as np
# import cupy as np
import matplotlib.pyplot as plt

def gen_mask(X):
    Row=X.shape[0]
    Colum=X.shape[1]
    mask=np.zeros((Row,Colum))
    for i in range(Row):
        for j in range(Colum):
            if X[i,j]>= np.max(X)/10:
                mask[i,j]=1
            else:
                mask[i,j]=0
    return mask


def data_test():
    # load data
    # path = "result/v8/ori_t_0.47_1000000_ideal_1_1.01_0.0006585_4.npy"
    # name="ori_t_0.47_1000000_ideal_1_1.01_0.0006585_4"

    path = "result/v9/ori_bs_0.47_10000_ideal_1_2_0.1_4.npy"
    name="ori_bs_0.47_10000_ideal_1_2_0.1_4.npy"
    ori_bs = np.load(path)

    # ori_bs = np.load("result/v7/ori_bs_0.47_1000000_evenly_0.25_1.01_0.0006585_4.npy")

    print("loading...completed!!")
    MM = np.zeros((100,100,4,4))
    m00 = ori_bs[:,:,0,0]
    # m01 = ori_bs[:,:,0,1]
    mask = gen_mask(m00)
    # fig = plt.figure()
    # plt.imshow(m00)
    # plt.show()

    fig = plt.figure()  
    for row in range(0,4):
        for colum in range(0,4):
            MM[:,:,row,colum]= ori_bs[:,:,row,colum]/m00

            plt.subplot(4,4,row*4+(colum+1)),plt.imshow(MM[:,:,row,colum] ,cmap='jet'),plt.title('M'+str(row)+str(colum))
            colorbar = plt.colorbar()
            plt.clim(-0.05,0.05)
            plt.subplot(4,4,1),plt.imshow(MM[:,:,0,0],cmap='jet'),plt.title('M00')
            colorbar = plt.colorbar()
            plt.clim(-0.01,0.01)
            plt.subplot(4,4,16),plt.imshow(MM[:,:,3,3],cmap='jet'),plt.title('M33')
            colorbar = plt.colorbar()
            plt.clim(-0.01,0.01)
    plt.show()
    fig.savefig('result/v9/figure_normalized_masked'+str(name)+'.png')
    # np.save('result/v8/normalized.png', MM)


    # mask = gen_mask(m00)
    # nor_m01 = m01/m00
    # print("normalizing...completed!!")
    # fig = plt.figure()
    # plt.subplot(1,4,1),plt.imshow(m00 ,cmap='jet'),plt.title("m00")
    # colorbar = plt.colorbar()
    # # plt.clim(-1,1)
    # plt.subplot(1,4,2),plt.imshow(m01 ,cmap='jet'),plt.title("m01")
    # colorbar = plt.colorbar()
    # # plt.clim(-1,1)
    # plt.subplot(1,4,3),plt.imshow(nor_m01 ,cmap='jet'),plt.title("nor_m01")
    # colorbar = plt.colorbar()
    # # i_max = np.max(m10)/np.max(m00)
    # # plt.clim(-i_max*2,i_max*2)
    # plt.clim(-0.1,0.1)

    # plt.subplot(1,4,4),plt.imshow(nor_m01*mask ,cmap='jet'),plt.title("masked_nor_m01")
    # # plt.clim(-0.1,0.1)
    # colorbar = plt.colorbar()
    # plt.show()
    # print("finished")


def principle_test():
    pass


if __name__ == "__main__":
    # Main program
    data_test()
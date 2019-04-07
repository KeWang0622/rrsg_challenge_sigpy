#!/usr/bin/env python
# Ke Wang 04/05/2019
# Contact: kewang@berkeley.edu
# Import package: Here we need to install sigpy
import h5py
import numpy as np
import sigpy.plot as pl
import sigpy as sp
import cfl
import matplotlib.pyplot as plt
from optparse import OptionParser



def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-l', '--lamda', dest='lamda',type='float',
                      default=0.1, help='lamda for l2 regularization')
    parser.add_option('-u', '--under', dest='under', type='int',
                      default=1, help='Undersampling rate for non-cartesian data')
    parser.add_option('-i', '--iteration', dest='iterations', type='int',
                      default=100, help='Iteration number for Conjugated Gradient Descent')
    
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    #read dataset: brain dataset
    #Here, the non-cartesian data consists of a trajectory coordinate file and a raw data file.
    h5_dataset_brain = h5py.File('rawdata_brain_radial_96proj_12ch.h5', 'r')
    h5_dataset_rawdata_name = list(h5_dataset_brain.keys())[0]
    h5_dataset_trajectory_name = list(h5_dataset_brain.keys())[1]
    trajectory = h5_dataset_brain.get(h5_dataset_trajectory_name).value
    rawdata = h5_dataset_brain.get(h5_dataset_rawdata_name).value


    gpu_number = 1
    if args.gpu:
        coor = trajectory[:2,:,:].transpose((1,2,0))
        coor = sp.backend.to_device(coor,gpu_number)
        rawdata_brain = rawdata[0,:,:,:].transpose((2,0,1))
        rawdata_brain = sp.backend.to_device(rawdata_brain,gpu_number)
        image = cfl.readcfl("img_igrid_brain")
        image_sos = sp.util.rss(image,3)
        sens_maps = image[:,:,0,:]/image_sos
        sens_maps_t = sens_maps.transpose((2,0,1))
        sens_maps_t = sp.backend.to_device(sens_maps_t,gpu_number)
    else:
        coor = trajectory[:2,:,:].transpose((1,2,0))
#         coor = sp.backend.to_device(coor,gpu_number)
        rawdata_brain = rawdata[0,:,:,:].transpose((2,0,1))
#         rawdata_brain = sp.backend.to_device(rawdata_brain,gpu_number)
        image = cfl.readcfl("img_igrid_brain")
        image_sos = sp.util.rss(image,3)
        sens_maps = image[:,:,0,:]/image_sos
        sens_maps_t = sens_maps.transpose((2,0,1))
#         sens_maps_t = sp.backend.to_device(sens_maps_t,gpu_number)
    
    # Subsample of the radial data
    #L_2 Normalization
    lamda_l2 = args.lamda
    proxg_l2 = sp.prox.L2Reg((1,300,300),lamda_l2)
    coor_subsample_2 = coor[:,::args.under,:]
    rawdata_brain_2 = rawdata_brain[:,:,::args.under]
    # pl.ScatterPlot(coor_subsample_2)
    S_2 = sp.linop.Multiply((1,300,300),sens_maps_t)
    NUFFT_2 = sp.linop.NUFFT((12,300,300),coor_subsample_2)
    Operator_2 = NUFFT_2*S_2
    img_rec_now_2 = sp.app.LinearLeastSquares(Operator_2,rawdata_brain_2,proxg=proxg_l2,max_iter=args.iterations).run()
    img_rec_now_2 = sp.backend.to_device(img_rec_now_2)
    im_abs_recon_now_2 = abs(img_rec_now_2[0,:,:])
    plt.figure(figsize=(15,8))
    plt.title("Reconstruction without l1-Wavelet Regularization (undersampled by factor %d)"%args.under)
    plt.imshow(im_abs_recon_now_2[::-1,:],cmap='gray')
    plt.show()
#!/usr/bin/python
#%%
import numpy as np
import os
import nibabel as nib
import time
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import cmaesForFWD
import ants
        
#%%
if __name__ == '__main__':
    print("start")

    segmentation = nib.load("/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/1 other patients/rec001_pre/segm.nii.gz").get_fdata()

    affine = nib.load("/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/1 other patients/rec001_pre/segm.nii.gz").affine

    pet = nib.load("/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/1 other patients/rec001_pre/Tum_FET.nii.gz").get_fdata()[:,:,:,0]

    WM= nib.load("/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/1 other patients/rec001_pre/WM.nii.gz").get_fdata()

    GM= nib.load("/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/1 other patients/rec001_pre/GM.nii.gz").get_fdata()

    #pet = pet * brainmask
    pet = pet / np.max(pet)


    #%%
    print("unique",np.unique(segmentation))

    FLAIR = segmentation == 3
    necrotic = segmentation == 4
    enhancing = segmentation == 1

    WM[segmentation >0] = 1

    #%%
    plt.imshow(WM[:, :, 75],  cmap="Greys")
    plt.imshow(FLAIR[:, :, 75],  cmap="Reds", alpha=0.5)
    plt.imshow(enhancing[:, :, 75],  cmap="Greens",alpha=0.5)
    plt.imshow(pet[:, :, 75],  cmap="Blues",alpha=0.5)

    #%%
    settings = {}
    # ranges from LMI paper with T = 100
    parameterRanges = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3], [0.5, 0.85], [0.001, 0.5]] 
    settings["parameterRanges"] = parameterRanges

    # init parameter
    settings["rho0"] = 0.02
    settings["dw0"] = 0.001

    settings["thresholdT1c"] = 0.675
    settings["thresholdFlair"] = 0.25
    
    # center of mass
    com = ndimage.measurements.center_of_mass(FLAIR)
    settings["NxT1_pct0"] = float(com[0] / np.shape(FLAIR)[0])
    settings["NyT1_pct0"] = float(com[1] / np.shape(FLAIR)[1])
    settings["NzT1_pct0"] = float(com[2] / np.shape(FLAIR)[2])

    settings["workers"] = 8#8#8#8#8
    settings["sigma0"] = 0.02    
    # if dir it changes with generations: key = from relative generations, value = resolution factor
    settings["resolution_factor"] ={ 0: 0.6, 0.333: 0.8, 0.6666: 1.0   }# 1# 0.6
    settings["generations"] = int(1500 /8) # 10000 samples

    solver = cmaesForFWD.CmaesSolver(settings, WM, GM, FLAIR, enhancing, pet, necrotic)
    resultTumor, resultDict = solver.run()

    # save results
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = "./resultsP001/"+ datetime +"_gen_"+ str(settings["generations"]) + "/"
    os.makedirs(path, exist_ok=True)
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    cmaesForFWD.writeNii(resultTumor, path = path+"result.nii.gz", affine = affine)




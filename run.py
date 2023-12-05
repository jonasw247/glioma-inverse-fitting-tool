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
        
#%%
if __name__ == '__main__':
    print("start")

    #GM = readNii("GM.nii.gz")#[::2, ::2, ::2]
    #WM = readNii("WM.nii.gz")#[::2, ::2, ::2]
    segmentation = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm006/preop/sub-tgm006_ses-preop_space-sri_seg.nii.gz").get_fdata()

    background = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm006/preop/sub-tgm006_ses-preop_space-sri_tissuemask.nii.gz").get_fdata()

    affine = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm006/preop/sub-tgm006_ses-preop_space-sri_tissuemask.nii.gz").affine

    brainmask =nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm006/preop/sub-tgm006_ses-preop_space-sri_brainmask.nii.gz").get_fdata()

    pet = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm006/preop/sub-tgm006_ses-preop_space-sri_fet.nii.gz").get_fdata()
    pet = pet * brainmask
    pet = pet / np.max(pet)
    #%%
    print("unique",np.unique(segmentation))

    FLAIR = segmentation > 0
    T1c = np.logical_or(segmentation == 1, segmentation == 4)

    GM = background == 2
    WM = background == 3

    plt.imshow(WM[:, :, 75],  cmap="Greys")
    plt.imshow(FLAIR[:, :, 75],  cmap="Reds", alpha=0.5)
    plt.imshow(T1c[:, :, 75],  cmap="Greens",alpha=0.5)
    plt.imshow(pet[:, :, 75],  cmap="Blues",alpha=0.5)

    #%%

    settings = {}
    # ranges from LMI paper with T = 100
    parameterRanges = [[0, 1], [0, 1], [0, 1], [0.0001, 0.225], [0.001, 3]] 
    settings["parameterRanges"] = parameterRanges

    # init parameter
    settings["rho0"] = 0.002
    settings["dw0"] = 0.001

    # center of mass
    com = ndimage.measurements.center_of_mass(FLAIR)
    settings["NxT1_pct0"] = float(com[0] / np.shape(FLAIR)[0])
    settings["NyT1_pct0"] = float(com[1] / np.shape(FLAIR)[1])
    settings["NzT1_pct0"] = float(com[2] / np.shape(FLAIR)[2])

    settings["workers"] = 8
    settings["sigma0"] = 0.02
    settings["generations"] = 1

    solver = cmaesForFWD.CmaesSolver(settings, WM, GM, FLAIR, T1c, pet)
    resultTumor, resultDict = solver.run()

    # save results
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = "./results/"+ datetime +"_gen_"+ str(settings["generations"]) + "/"
    os.makedirs(path, exist_ok=True)
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    cmaesForFWD.writeNii(resultTumor, path = path+"result.nii.gz", affine = affine)



# %%

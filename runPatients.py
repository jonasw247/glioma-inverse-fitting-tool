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
def run(edema, necrotic, enhancing, affine, pet, WM, GM, resultpath):
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
    com = ndimage.measurements.center_of_mass(edema)
    settings["NxT1_pct0"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct0"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct0"] = float(com[2] / np.shape(edema)[2])

    settings["workers"] = 9
    settings["sigma0"] = 0.02

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    settings["resolution_factor"] ={ 0: 0.6, 0.8: 0.8, 0.9: 1.0   }
    settings["generations"] = int(1000 /9) +1 # there are 9 samples in each step

    solver = cmaesForFWD.CmaesSolver(settings, WM, GM, edema, enhancing, pet, necrotic)
    resultTumor, resultDict = solver.run()

    # save results
    os.makedirs(resultpath, exist_ok=True)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    cmaesForFWD.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)

# 18 patients
if False: #__name__ == '__main__':

    for patientID in range(991,1001):
        try:
            dataPath = "/mnt/8tb_slot8/jonas/datasets/18_data/rec" + ("0000" + str(patientID))[-3:] + "_pre/"

            segmentation = nib.load(dataPath + "segm.nii.gz").get_fdata()

            affine = nib.load(dataPath + "segm.nii.gz").affine

            pet = nib.load(dataPath + "FET.nii.gz").get_fdata()
            if pet.ndim == 4:
                pet = pet[:,:,:,0]
            
            WM = nib.load(dataPath + "t1_wm.nii.gz").get_fdata()

            GM = nib.load(dataPath + "t1_gm.nii.gz").get_fdata()

            assert WM.shape == GM.shape == pet.shape == segmentation.shape

        except:
            print("patient not found ", patientID)
            continue

        #pet = pet * brainmask
        pet = pet / np.max(pet)

        #patient 006 has 2 and 3 for edema...
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 4
        enhancing = segmentation == 1

        #WM[segmentation >0] = 1

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/18_data/resultsP" + ("0000" + str(patientID))[-3:] + "/"
        run(edema, necrotic, enhancing, affine, pet, WM, GM, resultpath)

#tgm
if __name__ == '__main__':

    for patientID in range(41,100):
        try:
            dataPath = "/mnt/8tb_slot8/jonas/datasets/TGM/"

            segm = nib.load(dataPath + "tgm/tgm" +  ("0000" + str(patientID))[-3:] +"/preop/sub-tgm"+("0000" + str(patientID))[-3:]+"_ses-preop_space-sri_seg.nii.gz")

            

            segmentation = segm.get_fdata()
            affine = segm.affine

            tissuePath = dataPath + "preop_s3Results/tgm"+ ("0000" + str(patientID))[-3:] + "/"
            WM = nib.load(tissuePath + "sub-tgm"+ ("0000" + str(patientID))[-3:] +"_ses-preop_space-sri_t1_wm.nii.gz").get_fdata()
            
            GM = nib.load(tissuePath + "sub-tgm"+ ("0000" + str(patientID))[-3:] +"_ses-preop_space-sri_t1_gm.nii.gz").get_fdata()

        
        except:
            print("patient not found ", patientID)
            continue

        try:
            pet = nib.load(dataPath + "tgm/tgm" +  ("0000" + str(patientID))[-3:] +"/preop/sub-tgm"+("0000" + str(patientID))[-3:]+"_ses-preop_space-sri_fet.nii.gz").get_fdata()
            if pet.ndim == 4:
                pet = pet[:,:,:,0]
            #pet = pet * brainmask
            pet = pet / np.max(pet)
        except: # if no pet just set it to 0
            pet = np.zeros_like(WM)
        
        assert WM.shape == GM.shape == pet.shape == segmentation.shape

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        #WM[segmentation >0] = 1

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_results/" + ("0000" + str(patientID))[-3:] + "/"
        run(edema, necrotic, enhancing, affine, pet, WM, GM, resultpath)


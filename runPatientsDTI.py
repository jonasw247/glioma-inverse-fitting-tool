#!/usr/bin/python
#%%
import numpy as np
import os
import nibabel as nib
import time
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import cmaesDTI
import tools
from scipy.ndimage import binary_dilation
        
#%%
def run(edema, necrotic, enhancing, affine, diffusionTensors, csf, brainmask, resultpath):
    #%%
    settings = {}
    # fixed parameters that are not varied
    #TODO
    settings["fixedParameters"] = ["thresholdT1c",
                                    "thresholdFlair", "diffusionTensorExponent"]#,  "Dw","NxT1_pct", "NyT1_pct", "NzT1_pct"]

    # init parameter
    settings["rho"] = 3.0
    settings["Dw"] = 0.15
    settings["diffusionEllipsoidScaling"] = 10
    settings["diffusionTensorExponent"] = 1.0
    settings["thresholdT1c"] = 0.9
    settings["thresholdFlair"] = 0.25

    # center of mass
    com = ndimage.measurements.center_of_mass(edema)
    settings["NxT1_pct"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct"] = float(com[2] / np.shape(edema)[2])

    # set parameter ranges
    settings["rho_range"] = [0.001, 5.0]
    settings["Dw_range"] = [0.001, 5.0]
    settings["thresholdT1c_range"] = [0.5, 0.9]
    settings["thresholdFlair_range"] = [0.001, 0.5]
    settings["NxT1_pct_range"] = [0,1]
    settings["NyT1_pct_range"] = [0,1]
    settings["NzT1_pct_range"] = [0,1]
    settings["diffusionEllipsoidScaling_range"] = [0.1, 100.0]
    settings["diffusionTensorExponent_range"] = [0.1, 10.0]

    # algorithm settings
    settings["workers"] =9 #9# 1#9 #9#4 # 9
    settings["sigma0"] = 0.06
    settings["lossLambdaT1"] = 0.2
    settings["lossLambdaFlair"] = 0.8

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    #TODO
    settings["resolution_factor"] = 0.1#{ 0: 0.5, 0.3:0.6, 0.8: 0.8, 0.9: 1.0}
    settings["generations"] =  1#int(1000 /9) +1 # there are 9 samples in each step

    solver = cmaesDTI.CmaesSolver(settings, csf, diffusionTensors, edema, enhancing, necrotic)
    resultTumor, resultDict = solver.run()

    # save results
    os.makedirs(resultpath, exist_ok=True)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    tools.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)
    
    print("Done For This Patient")

# 18 patients old!!
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
if  __name__ == '__main__':

    patients = [51, 16,  31, 42,  1 , 2 ,3,4,5,6,7,8,9,10, 11, 12, 13] # 
    #patients = np.arange(20,40,1)
    print(patients)
    for patientID in patients:
        try:
            dataPath = "/mnt/8tb_slot8/jonas/datasets/TGM/"

            dtiPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/rgbResults/"

            segm = nib.load(dataPath + "tgm/tgm" +  ("0000" + str(patientID))[-3:] +"/preop/sub-tgm"+("0000" + str(patientID))[-3:]+"_ses-preop_space-sri_seg.nii.gz")

            segmentation = segm.get_fdata()
            affine = segm.affine

            brainTissue = nib.load(dataPath + "tgm/tgm" +  ("0000" + str(patientID))[-3:] +"/preop/sub-tgm"+("0000" + str(patientID))[-3:]+"_ses-preop_space-sri_tissuemask.nii.gz")

            brainmask = nib.load(dataPath + "tgm/tgm" +  ("0000" + str(patientID))[-3:] +"/preop/sub-tgm"+("0000" + str(patientID))[-3:]+"_ses-preop_space-sri_brainmask.nii.gz").get_fdata()

            diffusionTensors = nib.load(dtiPath + "sub-tgm"+ ("0000" + str(patientID))[-3:] +"_ses-preop_space-sri_dti_tensor.nii.gz").get_fdata()

        except:
            print("patient not found ", patientID)
            continue

        
        csf = binary_dilation(brainTissue == 1, iterations = 1)

        diffusionTensors[csf] = 0

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results_12initLargerhoandExponent/" + ("0000" + str(patientID))[-3:] + "/"

        #TODO
        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results_testing/" + ("0000" + str(patientID))[-3:] + "/"

        #TODO
        run(edema, necrotic, enhancing, affine, diffusionTensors, csf, brainmask, resultpath)

#respond old!!
if False: # __name__ == '__main__':
    for patientID in range(120,130):
        try:
            # save the parameters and the tumor
            patientNumber = ("000000" + str(patientID))[-3:]
            print("patient number: ", patientNumber)

            patientPath = "/mnt/8tb_slot8/jonas/datasets/ReSPOND/respond/respond_tum_"+ patientNumber+"/d0/"
            
            segmentationNiiPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_seg.nii.gz"
            segm = nib.load(segmentationNiiPath)

            segmentation = segm.get_fdata()
            affine = segm.affine

            tissuePath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_tissuemask.nii.gz"
            tissue = nib.load(tissuePath).get_fdata()

        except:
            print("patient not found ", patientID)
            continue

        try:
            petPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_fet.nii.gz"
            pet = nib.load(petPath).get_fdata()
            if pet.ndim == 4:
                pet = pet[:,:,:,0]
            #pet = pet * brainmask
            pet = pet / np.max(pet)
        except: # if no pet just set it to 0
            pet = np.zeros_like(tissue)

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        WM, GM = segmentation *0.0, segmentation *0.0
        WM[tissue == 3] = 1.0
        GM[tissue == 2] = 1.0

        WM[segmentation >0] = 1.0

        assert WM.shape == GM.shape == pet.shape == segmentation.shape

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        resultpath = '/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results/' + str(patientNumber) + 'newSettings/'
        run(edema, necrotic, enhancing, affine, pet, WM, GM, resultpath)
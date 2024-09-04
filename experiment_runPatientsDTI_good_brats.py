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
import TumorGrowthToolkit.FK_DTI.tools as toolsDTI
import gc
import importlib
import subprocess
import sys
from multiprocessing import Pool, cpu_count

doLog = True

        
#%%
def run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, resultpath):
    
    settings = {}
    # fixed parameters that are not varied
    #TODO
    # only optimize origin, rho and final volume for now
    settings["fixedParameters"] = ["Dw", "diffusionEllipsoidScaling", "diffusionTensorExponent",  "stopping_time", "thresholdT1c", "thresholdFlair" ]#,,  "Dw","NxT1_pct", "NyT1_pct", "NzT1_pct"],

    # init parameter
    settings["rho"] = 0.8
    settings["Dw"] = 1.0
    settings["diffusionEllipsoidScaling"] = 1
    settings["diffusionTensorExponent"] = 1
    settings["thresholdT1c"] = 0.9
    settings["thresholdFlair"] = 0.5
    settings["stopping_volume"] = 0.7*(np.sum(edema) + np.sum(necrotic) + np.sum(enhancing))
    settings["stopping_time"] = 10000000

    # center of mass
    com = ndimage.center_of_mass(edema)
    settings["NxT1_pct"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct"] = float(com[2] / np.shape(edema)[2])

    # set parameter ranges
    settings["rho_range"] = [0.01, 5.0]
    settings["Dw_range"] = [0.001, 5.0]
    settings["thresholdT1c_range"] = [0.5, 0.9]
    settings["thresholdFlair_range"] = [0.01, 0.5]
    settings["NxT1_pct_range"] = [0,1]
    settings["NyT1_pct_range"] = [0,1]
    settings["NzT1_pct_range"] = [0,1]
    settings["diffusionEllipsoidScaling_range"] = [0.1, 100.0]
    settings["diffusionTensorExponent_range"] = [0.1, 10.0]
    settings["stopping_volume_range"] = [100, np.sum(brainmask)]
    settings["stopping_time_range"] = [0, 1000000000]

    # algorithm settings
    settings["workers"] = 0#9#0 #9# 1#9 #9#4 # 9
    settings["sigma0"] = 0.01
    settings["lossLambdaT1"] = 0.5 #0.2
    settings["lossLambdaFlair"] = 0.5 # 0.8

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    settings["resolution_factor"] = { 0: 0.5, 0.6: 0.6, 0.7:0.7, 0.85:0.85, 0.9: 0.9, 0.95: 1.0} # 0.5 #{ 0: 0.5, 0.75: 0.6, 0.85:0.8, 0.9: 0.8, 0.95: 1.0}
    settings["generations"] = 30 #125#TODO int(1000 /9) +1 # there are 9 samples in each step

    solver = cmaesDTI.CmaesSolver(settings, diffusionTensors, edema, enhancing, necrotic)
    resultTumor, resultDict = solver.run()

    # save results
    os.makedirs(resultpath, exist_ok=True)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    nibImg = nib.Nifti1Image(resultTumor, affine)
    nib.save(nibImg, resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz")

    tools.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)
    
    del solver, resultTumor, resultDict, nibImg
    print("Done For This Patient")

#%%
def process_patient(patientID):

    patString = ("000000" + str(patientID))[-5:]
    try:
        segmPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_{patString}/preop/sub-BraTS2021_{patString}_ses-preop_space-sri_seg.nii.gz"
        dtiPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{patString}/transformed_reoriented_tensor.nii.gz"
        tissuePath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{patString}/transformed_tissue.nii.gz"

        segm = nib.load(segmPath)
        segmentation = segm.get_fdata()
        affine = segm.affine

        brainTissue = nib.load(tissuePath).get_fdata()
        diffusionTensorsLower = nib.load(dtiPath).get_fdata()[:, :, :, 0, :]
        diffusionTensors = toolsDTI.get_tensor_from_lower6(diffusionTensorsLower)

        print("found data for patient", patientID)
    except Exception as e:
        print(f"patient {patientID} not found: {e}")
        return

    CSFMask = brainTissue == 1
    CSFMask[segmentation > 0] = 0
    diffusionTensors[CSFMask] = 0
    brainmask = brainTissue > 0

    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4

    if np.sum(edema) + np.sum(necrotic) + np.sum(enhancing) < 25 or np.sum(edema) < 25:
        print("Too small tumor for patient", patientID)
        return

    resultpath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_DTI_results_02/BraTS2021_{('0000' + str(patientID))[-3:]}/"

    run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, resultpath)

    # Explicit cleanup
    del segm, segmentation, brainTissue, diffusionTensorsLower, diffusionTensors
    gc.collect()
if __name__ == '__main__':
    process_patient(115)

#%%

if False: #__name__ == '__main__':
    if len(sys.argv) > 1:
        patientID = sys.argv[1]
        process_patient(patientID)
    else:
        #NOT WORKING
        patients =np.arange(0, 180, 1) #  [115]#
        print(patients)
        
        # Determine the number of worker processes to use
        num_workers = 2

        #for patient in patients:
        #    process_patient(patient)
            
        with Pool(processes=num_workers) as pool:
            for patientID in patients:
                pool.apply(process_patient, args=(patientID,))
            
    try:
        
        print("Done for patient", patientID)
    except Exception as e:
        print(f"Error processing patient {patientID}: {e}")
#%%

#%%
if False:#__name__ == '__main__':
    #patientID = sys.argv[1] # This is done  as the starting volume is somehow 0 for the second iteration in the loop
    patients = np.arange(113, 400, 1)
    print(patients)
    for patientID in patients:
        process_patient(patientID)
        print("Done for patient", patientID)
#%%
if False:# __name__ == '__main__':

    #patients = [51, 16,  31, 42,  1 , 2 ,3,4,5,6,7,8,9,10, 11, 12, 13] # 
    patients = np.arange(80,400,1)
    #patients = [16]
    print(patients)
    for patientID in patients:
        patString = ("000000" + str(patientID))[-5:]
        try:
            segmPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_"+patString+"/preop/sub-BraTS2021_"+patString+"_ses-preop_space-sri_seg.nii.gz"

            dtiPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_"+patString+"/transformed_reoriented_tensor.nii.gz"
            tissuePath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_"+patString+"/transformed_tissue.nii.gz"

            segm = nib.load(segmPath)
            segmentation = segm.get_fdata()
            affine = segm.affine

            brainTissue = nib.load(tissuePath).get_fdata()
            diffusionTensorsLower = nib.load(dtiPath).get_fdata()[:,:,:,0,:]
            diffusionTensors = toolsDTI.get_tensor_from_lower6(diffusionTensorsLower)

            print("found data for patient" , patientID)
        

        except:
            print("patient not found ", patientID)

            continue

        #CSFMask = binary_dilation(brainTissue == 1, iterations = 1)
        CSFMask = brainTissue == 1
        CSFMask[segmentation >0] = 0

        diffusionTensors[CSFMask] = 0

        brainmask = brainTissue >0


        """plt.imshow((diffusionTensors/np.max(diffusionTensors))[:,:,70,:,1])
        plt.show()
        plt.imshow(CSFMask[:,:,70])"""

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_DTI_results_run01/BraTS2021_" + ("0000" + str(patientID))[-3:] + "/"

        resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_DTI_results_testing/BraTS2021_" + ("0000" + str(patientID))[-3:] + "/"
        #TODO
        #resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results_testing/" + ("0000" + str(patientID))[-3:] + "/"

        run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, resultpath)
        gc.collect()
# %%

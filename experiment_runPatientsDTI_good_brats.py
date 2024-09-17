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
experimentName = "18_testDTI"#"17_testDTIexponent" #"15_testDTI"# "12_testDTI_highRes"#
debug = False # TODOCheck
if debug:
    experimentName += "debug"
    doLog = False
        
#%%
def run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, resultpath, gm, wm, runName = "run"):
    
    settings = {}

    if "DTI" in experimentName:
        settings["runNormalFKInsteadOfDTI"] = False 
    else:
        settings["runNormalFKInsteadOfDTI"] = True
    if settings["runNormalFKInsteadOfDTI"]:
        print("Attention -----------------")
        print("running normal FK")
        print("Attention -----------------")

    # fixed parameters that are not varied
    # only optimize origin, rho and final volume for now
    settings["fixedParameters"] = [ "diffusionEllipsoidScaling",   "stopping_time",  "thresholdT1c", "thresholdFlair","rho"]#"diffusionTensorExponent",,"Dw",,"NxT1_pct", "NyT1_pct", "NzT1_pct"], , "thresholdFlair", 

    # init parameter
    settings["rho"] = 0.5 #0.5#0.1 # TODO
    settings["Dw"] = 5.0
    settings["RatioDw_Dg"] = 10

    settings["diffusionEllipsoidScaling"] = 1
    settings["diffusionTensorExponent"] = 1
    settings["thresholdT1c"] = 0.66
    settings["thresholdFlair"] = 0.33
    settings["stopping_volume"] = 0.7*(np.sum(edema) + np.sum(necrotic) + np.sum(enhancing))
    settings["stopping_time"] = 10000000


    # center of mass
    com = ndimage.center_of_mass(necrotic + enhancing)
    settings["NxT1_pct"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct"] = float(com[2] / np.shape(edema)[2])


    # set parameter ranges
    settings["rho_range"] = [0.01, 5.0]
    settings["Dw_range"] = [0.001, 120.0] #TODO
    settings["RatioDw_Dg_range"] = [0.1, 100.0]
    settings["thresholdT1c_range"] = [0.5, 0.9]
    settings["thresholdFlair_range"] = [0.01, 0.5]
    settings["NxT1_pct_range"] = [0,1]
    settings["NyT1_pct_range"] = [0,1]
    settings["NzT1_pct_range"] = [0,1]
    settings["diffusionEllipsoidScaling_range"] = [0.1, 100.0]
    settings["diffusionTensorExponent_range"] = [0.1, 10.0]
    settings["stopping_volume_range"] = [0.1 * (np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)), np.sum(brainmask) /2]
    settings["stopping_time_range"] = [0, 1000000000]


    # algorithm settings
    settings["workers"] =0 #9# 9#9#0#9#0 #9# 1#9 #9#4 # 9 TODO
    settings["sigma0"] = 0.02 # TODO
    weighLossByVolume = True
    settings["weighLossByVolume"] = weighLossByVolume
    if weighLossByVolume:
        volumeCore = np.sum(necrotic) + np.sum(enhancing)
        volumeEdema = np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)
        totalVolume = volumeCore + volumeEdema
        relVolumeCore = volumeCore/ totalVolume
        relVolumeEdema = 1 - relVolumeCore
        print("relVolumeCore", relVolumeCore)
        settings["lossLambdaT1"] = relVolumeCore
        settings["lossLambdaFlair"] = relVolumeEdema
        print("rel core volume:", relVolumeCore, "rel edema volume:", relVolumeEdema)
    else: #TODO
        settings["lossLambdaT1"] = 0 #0.5 #0.2
        settings["lossLambdaFlair"] = 1 #0.5 # 0.8

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    settings["resolution_factor"] = 0.5#{ 0: 0.5, 0.7: 0.6, 0.8:0.7, 0.85:0.8, 0.9: 0.9, 0.95: 1.0} # 0.5 #{ 0: 0.5, 0.75: 0.6, 0.85:0.8, 0.9: 0.8, 0.95: 1.0}
    settings["generations"] = 102 #101 #125#TODO int(1000 /9) +1 # there are 9 samples in each step
    if debug:
        settings["generations"] = 40
        resolution_factor = 0.5

    solver = cmaesDTI.CmaesSolver(settings, diffusionTensors, edema, enhancing, necrotic, gm, wm, logNameProject = "evolutionary_sampling" + experimentName, logNameRun = runName)
    resultTumor, resultDict = solver.run()

    # save results
    resPathFolder = resultpath# os.path.join(resultpath.split("/")[0:-1])
    os.makedirs(resPathFolder, exist_ok=True)

    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    nibImg = nib.Nifti1Image(resultTumor, affine)
    nib.save(nibImg, resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz")

    tools.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)
    
    del solver, resultTumor, resultDict, nibImg
    print("DTI done for this Patient")
    gc.collect()

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

    wm = brainTissue == 3
    gm = brainTissue == 2

    #exclude CSF
    CSFMask = brainTissue == 1
    # include tumor segmentation region,
    mask = CSFMask.copy()
    mask[segmentation > 0] = 0
    diffusionTensors[mask] = 0

    #exclude CSF 
    wm[mask] = 0
    gm[mask] = 0
    gm[np.logical_and(CSFMask, segmentation>0)] = True

    brainmask = brainTissue > 0

    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4

    if np.sum(edema) + np.sum(necrotic) + np.sum(enhancing) < 25 or np.sum(edema) < 25:
        print("Too small tumor for patient", patientID)
        return

    resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_"+experimentName+f"/BraTS2021_{('0000000' + str(patientID))[-5:]}/sub-BraTS2021_{('0000000' + str(patientID))[-5:]}_ses-preop_space-sri_"

    run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, resultpath, gm, wm, runName = f"/BraTS2021_{('0000000' + str(patientID))[-5:]}")

    # Explicit cleanup
    del segm, segmentation, brainTissue, diffusionTensorsLower, diffusionTensors
    gc.collect()


#%%

if False:# __name__ == '__main__':
    if len(sys.argv) > 1:
        patientID = sys.argv[1]
        process_patient(patientID)
    else:
        process_patient(14)



if False: #True: #__name__ == '__main__':
    process_patient(16) #TODO

    for i in range(14, 160):
        break # TODO
        try:
            process_patient(i)
        except Exception as e:
            print(f"Error processing patient {i}: {e}")
#%%

def try_process_patient(patient):
    try:
        process_patient(patient)
    except Exception as e:
        print(f"Error processing patient {patient}: {e}")


if True:
    #patientList = [238, 250, 246, 263, 364]
    with Pool(5) as p:
        p.map(try_process_patient, range(14, 300))
        
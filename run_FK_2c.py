#!/usr/bin/python
# %%
import numpy as np
import os
import nibabel as nib
import time
from scipy import ndimage
import matplotlib.pyplot as plt
import FK_2c_cmaes

# %%
if __name__ == '__main__':
    print("start")

    # Load data
    segmentation = nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/segm.nii.gz").get_fdata()
    pet = nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/FET.nii.gz").get_fdata()

    # Extract tumor regions
    FLAIR = segmentation == 3
    necrotic = segmentation == 4
    enhancing = segmentation == 1

    # Get probability images for Grey and White matter
    GM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/gm_data.nii.gz").get_fdata()
    WM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/wm_data.nii.gz").get_fdata()

    # Visualization
    plt.imshow(WM[:, :, 75], cmap="Greys",alpha=0.5)
    plt.imshow(FLAIR[:, :, 75], cmap="Reds", alpha=0.5)
    plt.imshow(enhancing[:, :, 75], cmap="Greens", alpha=0.5)
    plt.imshow(necrotic[:, :, 75], cmap="Blues", alpha=0.5)
    plt.show()
    
    # Visualization
    plt.imshow(pet[:, :, 75], cmap="Blues", alpha=0.5)
    plt.show()
    


    # %%
    # Configuration settings
    settings = {}
    # Define parameter ranges for the new model
    #parameterRanges = [
    #    [0, 1], [0, 1], [0, 1], [0.1670243, 0.1670245], [0.513545, 0.513546], [0.41054551, 0.41054551], [0.06375344, 0.06375344], [3.0144, 3.0144], [0.3977, 0.3977],
    #    [0.1800172864753556, 0.1800172864753557], [0.2930917239803693, 0.2930917239803694], [0.06983855590510551, 0.06983855590510552]
    #]
    parameterRanges = [
            [0, 1], [0, 1], [0, 1], [0.1, 1], [0.1, 5], [0.1, 0.8], [0.01, 0.8], [0.1, 5], [0.1, 1],
            [0.10, 0.25], [0.3, 0.7], [0.01, 0.2]
        ]
    settings["parameterRanges"] = parameterRanges


    # Initialize parameters
    settings["rho0"] = 0.1670243
    settings["dw0"] = 0.513545
    # Add initial values for new parameters
    settings["lambda_np0"] = 0.41054551
    settings["sigma_np0"] = 0.06375344
    settings["D_s0"] = 3.0144
    settings["lambda_s0"] = 0.3977

    # Thresholds and center of mass
    settings["thresholdFlair0"] = 0.1800172864753556
    settings["thresholdT1c0"] = 0.2930917239803693
    settings["thresholdNecro0"] = 0.06983855590510551  # New threshold for necrosis

    com = ndimage.measurements.center_of_mass(necrotic)
    settings["NxT1_pct0"] = float(com[0] / np.shape(necrotic)[0])
    settings["NyT1_pct0"] = float(com[1] / np.shape(necrotic)[1])
    settings["NzT1_pct0"] = float(com[2] / np.shape(necrotic)[2])

    settings["workers"] = 8
    settings["sigma0"] = 0.02
    settings["generations"] = 20

    # Create solver instance and run
    solver = FK_2c_cmaes.CmaesSolver(settings, WM, GM, FLAIR, enhancing, pet, necrotic)
    resultTumor, resultDict = solver.run()

    # Save results
    datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = "./results/" + datetime + "_gen_" + str(settings["generations"]) + "/"
    os.makedirs(path, exist_ok=True)
    np.save(path + "settings.npy", settings)
    np.save(path + "results.npy", resultDict)
    FK_2c_cmaes.writeNii(resultTumor, base_path=path)

# %%
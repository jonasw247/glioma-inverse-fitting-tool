#!/usr/bin/python
# %%
import numpy as np
import os
import nibabel as nib
import time
from scipy import ndimage
import matplotlib.pyplot as plt
import FK_2c_cmaes
import argparse

# %%
if __name__ == '__main__':
    print("start")

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process base_path.')
    parser.add_argument('base_path', type=str, help='Base path for the data files')
    args = parser.parse_args()

    # Use the base_path from the argument
    base_path = args.base_path

    # Load data using os.path.join to concatenate the base path with the file names
    segmentation = nib.load(os.path.join(base_path, "segm.nii.gz")).get_fdata()
    pet = nib.load(os.path.join(base_path, "FET.nii.gz")).get_fdata()

    try:
        # Attempt to load the first set of files
        GM = nib.load(os.path.join(base_path, "gm_data.nii.gz")).get_fdata()
        WM = nib.load(os.path.join(base_path, "wm_data.nii.gz")).get_fdata()
    except FileNotFoundError:
        # If the first set of files is not found, load the alternative files
        GM = nib.load(os.path.join(base_path, "t1_gm.nii.gz")).get_fdata()
        WM = nib.load(os.path.join(base_path, "t1_wm.nii.gz")).get_fdata()


    # Extract tumor regions
    FLAIR = segmentation == 3
    necrotic = segmentation == 4
    enhancing = segmentation == 1


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
    settings["rho0"] = 0.3
    settings["dw0"] = 0.3
    # Add initial values for new parameters
    settings["lambda_np0"] = 0.5
    settings["sigma_np0"] = 0.05
    settings["D_s0"] = 1.0144
    settings["lambda_s0"] = 0.3

    # Thresholds and center of mass
    settings["thresholdFlair0"] = 0.1
    settings["thresholdT1c0"] = 0.4
    settings["thresholdNecro0"] = 0.05  # New threshold for necrosis

    com = ndimage.measurements.center_of_mass(necrotic)
    settings["NxT1_pct0"] = float(com[0] / np.shape(necrotic)[0])
    settings["NyT1_pct0"] = float(com[1] / np.shape(necrotic)[1])
    settings["NzT1_pct0"] = float(com[2] / np.shape(necrotic)[2])

    settings["workers"] = 16
    settings["sigma0"] = 0.02
    settings["generations"] = 502
    
    # if dir it changes with generations: key = from relative generations, value = resolution factor
    #settings["resolution_factor"] ={0: 0.42, 0.6: 0.55, 0.9: 0.65}
    settings["resolution_factor"] = 0.55

    
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
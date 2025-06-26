#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
import trimesh

dtiPath ="/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_28_testDTIFK_init_larger_std_butterfly/"
fkPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_28_testDTIFKOnly_init_larger_std_butterfly/"

#evolutionary_sampling18_testFK_butterfly

patID = 246
originalTumorLocationPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_{str(patID).zfill(5)}/preop/"

originalSegmentation = nib.load(originalTumorLocationPath + f"sub-BraTS2021_{str(patID).zfill(5)}_ses-preop_space-sri_seg.nii.gz").get_fdata()
#gm_tissue = nib.load(tissuePath).get_fdata()

#sub-BraTS2021_00014_ses-preop_space-sri_gen_101_result.nii.gz
fkPrediction =  nib.load(fkPath + f"BraTS2021_{str(patID).zfill(5)}/sub-BraTS2021_{str(patID).zfill(5)}_ses-preop_space-sri_gen_101_result.nii.gz").get_fdata()
dtiPrediction = nib.load(dtiPath + f"BraTS2021_{str(patID).zfill(5)}/sub-BraTS2021_{str(patID).zfill(5)}_ses-preop_space-sri_gen_101_result.nii.gz").get_fdata()

tissuePath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{str(patID).zfill(5)}/transformed_tissue.nii.gz"
rgbPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{str(patID).zfill(5)}/transformed_reoriented_tensor_rgb.nii.gz"
brainTissue = nib.load(tissuePath).get_fdata()
rgbTensors = nib.load(rgbPath).get_fdata()[:, :, :, 0, :]
z = 35
centerOfMass = np.array(np.where(originalSegmentation > 0)).mean(axis=1).astype(int)
x,y,z = centerOfMass

x,y,z = 160,160, 74
plt.figure(figsize=(6, 6))
plt.imshow(brainTissue[:, :, z], cmap="gray", alpha=1)
plt.imshow(fkPrediction[:, :, z], cmap="hot", alpha=dtiPrediction[:, :, z], vmin=0, vmax=1)
plt.imshow(originalSegmentation[:, :, z], alpha=(originalSegmentation[:, :, z] > 0) * 0.4, cmap="Greens")
plt.xlabel("Slice 50")
plt.ylabel("Slice 50")
plt.title("Fk Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()
#%%
rgbTensors = rgbTensors # Remove the last dimension if it is not needed**2


plt.figure(figsize=(6, 6))
plt.imshow(rgbTensors[:, :, z,:], cmap="gray", alpha=1)
#plt.imshow(dtiPrediction[:, :, z], cmap="hot", alpha=0.5, vmin=0, vmax=1)
#plt.imshow(originalSegmentation[:, :, z], alpha=(originalSegmentation[:, :, z] > 0) * 0.4, cmap="Greens")
plt.xlabel("Slice 50")
plt.ylabel("Slice 50")
plt.title("DTI Prediction")
plt.axis("off")
plt.tight_layout()
plt.show()

#%%

# Normalize the last dimension (RGB vector) to unit length
def getCoolLookingRGB(rgbTensors):
    norm = np.linalg.norm(rgbTensors, axis=-1, keepdims=True)
    std = np.std(rgbTensors, axis=-1, keepdims=True)
    norm[norm == 0] = 1  # avoid division by zero
    rgbTensorsPlot = rgbTensors / norm  * std  # Normalize and scale by standard deviation
    rgbTensorsPlot /= np.max(rgbTensorsPlot)
    return rgbTensorsPlot**0.7

#means = np.mean(rgbTensors, axis=2)
#rgbTensorsPlot = rgbTensors- means + 0.5 # Adjust brightness

rgbTensorsPlot = getCoolLookingRGB(rgbTensors)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# --- DTI Prediction on RGB Tensors ---
#%%
stops = [
    (0.0,  'red'),   # 0   → black
    #(0.1,  'red'),     # 0.1 → red
    (0.6,  'yellow'),  # 0.4 → yellow
    (1.0,  'white'),   # 1   → white
]
alphaLim = 0.01
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list('black_red_yellow_white', stops)
# Axial (z)
plt.figure(figsize=(6, 6))
plt.imshow(rgbTensors[:, :, z, :], alpha=1)
plt.imshow(dtiPrediction[:, :, z], cmap=custom_cmap, alpha=(dtiPrediction[:, :, z]> 0.01)*dtiPrediction[:, :, z]**0.5, vmin=0, vmax=1)
plt.imshow(originalSegmentation[:, :, z], alpha=(originalSegmentation[:, :, z] > 0) * 0.4, cmap="Greens")
plt.title("DTI Prediction on RGB Tensors - Axial (z)")
plt.axis("off")
plt.tight_layout()
plt.show()

#%%
# Coronal (y)
plt.figure(figsize=(6, 6))
plt.imshow(rgbTensors[:, y, :, :], alpha=1)
plt.imshow(dtiPrediction[:, y, :], cmap=custom_cmap, alpha= (dtiPrediction[:, y, :]> 0.01)*dtiPrediction[:, y, :]**0.5, vmin=0, vmax=1)
#plt.imshow(originalSegmentation[:, y, :], alpha=(originalSegmentation[:, y, :] > 0) * 0.4, cmap="Greens")
plt.title("DTI Prediction on RGB Tensors - Coronal (y)")
plt.axis("off")
plt.tight_layout()
plt.show()

"""# Sagittal (x)
plt.figure(figsize=(6, 6))
plt.imshow(rgbTensorsPlot[x, :, :, :], alpha=1) 
plt.imshow(dtiPrediction[x, :, :], cmap="hot", alpha=dtiPrediction[x, :, :], vmin=0, vmax=1)
plt.imshow(originalSegmentation[x, :, :], alpha=(originalSegmentation[x, :, :] > 0) * 0.4, cmap="Greens")
plt.title("DTI Prediction on RGB Tensors - Sagittal (x)")
plt.axis("off")
plt.tight_layout()
plt.show()
"""
# --- FK Prediction on Brain Tissue ---
#%%
# Axial (z)
plt.figure(figsize=(6, 6))
plt.imshow(brainTissue[:, :, z], cmap="gray", alpha=1)
plt.imshow(fkPrediction[:, :, z], cmap="hot", alpha=fkPrediction[:, :, z], vmin=0, vmax=1)
plt.imshow(originalSegmentation[:, :, z], alpha=(originalSegmentation[:, :, z] > 0) * 0.4, cmap="Greens")
plt.title("FK Prediction - Axial (z)")
plt.axis("off")
plt.tight_layout()
plt.show()


# Coronal (y)
plt.figure(figsize=(6, 6))
plt.imshow(brainTissue[:, y, :], cmap="gray", alpha=1)
plt.imshow(fkPrediction[:, y, :], cmap="hot", alpha=fkPrediction[:, y, :], vmin=0, vmax=1)
plt.imshow(originalSegmentation[:, y, :], alpha=(originalSegmentation[:, y, :] > 0) * 0.4, cmap="Greens")
plt.title("FK Prediction - Coronal (y)")
plt.axis("off")
plt.tight_layout()
plt.show()
"""
# Sagittal (x)
plt.figure(figsize=(6, 6))
plt.imshow(brainTissue[x, :, :], cmap="gray", alpha=1)
plt.imshow(fkPrediction[x, :, :], cmap="hot", alpha=fkPrediction[x, :, :], vmin=0, vmax=1)
plt.imshow(originalSegmentation[x, :, :], alpha=(originalSegmentation[x, :, :] > 0) * 0.4, cmap="Greens")
plt.title("FK Prediction - Sagittal (x)")
plt.axis("off")
plt.tight_layout()
plt.show()"""
#%%




fkPathRuns = os.listdir(fkPath)
dtiPathRuns = os.listdir(dtiPath)   


patsDTI, patsFK, fks, dtis = [], [], [], []
for pat in range(14, 3000):
    patstring = f"BraTS2021_{str(pat).zfill(5)}"

    if os.path.exists(dtiPath + patstring):
        dirs = os.listdir(dtiPath + patstring)
        for d in dirs:
            if "100_results.npy" in d or "101_results.npy" in d or "102_results.npy" in d or "50_results.npy" in d:
                dtis.append(np.load(f"{dtiPath}{patstring}/{d}", allow_pickle=True).item())
                patsDTI.append(pat)               


    if os.path.exists(fkPath + patstring):
        dirs = os.listdir(fkPath + patstring)
        for d in dirs:
            if "100_results.npy" in d or "101_results.npy" in d or "102_results.npy" in d or "50_results.npy" in d:
                fks.append(np.load(f"{fkPath}{patstring}/{d}", allow_pickle=True).item())
                patsFK.append(pat)

patsFK = np.array(patsFK).astype(str)
patsDTI = np.array(patsDTI).astype(str)
       
#%%
def getLossOfRun(res):
    lossDir = res["lossDir"]
    minLoss = 1
    for i in range(len(lossDir)):
        for j in range(len(lossDir[i])):
            if lossDir[i][j]["lossTotal"] < minLoss:
                minLoss = lossDir[i][j]["lossTotal"]
                diceFlair = lossDir[i][j]["diceFlair"]
                diceT1c = lossDir[i][j]["diceT1c"]
                opt = lossDir[i][j]["allParams"]
                bestLossDir = lossDir[i][j]

    return minLoss, opt, bestLossDir, diceFlair, diceT1c

def getLossForListOfRuns(runs):
    losses, diceFlairs, diceT1cs = [], [], []
    for run in runs:
        minLoss, opt, bestLossDir, diceFlair, diceT1c = getLossOfRun(run)
        losses.append(minLoss)
        diceFlairs.append(diceFlair)
        diceT1cs.append(diceT1c)
        
    return losses,diceFlairs, diceT1cs 

def getValueForListOfRuns(runs, key):
    values = []
    for run in runs:
        values.append(run[key])
    return values



def getRuntimeforListOfRuns(runs):
    runtimes = []
    for run in runs:
        runtimes.append(run["time_min"])
    return runtimes

# %% get final volumes# %%

def getSphericity(patientsNames):
    sphericitys, volumes = [], []
    for name in patientsNames:
        patString = ("000000" + name)[-5:]
        segmentation = nib.load("/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_"+ patString+ "/preop/sub-BraTS2021_"+patString +"_ses-preop_space-sri_seg.nii.gz").get_fdata()
        volume = np.sum(segmentation > 0)


        verts, faces, normals, values = measure.marching_cubes(segmentation>0, level=0)

        # Create a mesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Calculate the surface area
        tumor_surface_area = mesh.area
        sphericity = np.pi **(1/3)  *  (6 * volume) ** (2/3) / tumor_surface_area
        sphericitys.append(sphericity)
        volumes.append(volume)
    return sphericitys, volumes
#%%
lossDTIs, diceFlairDTIs, diceT1cDTIs = getLossForListOfRuns(dtis)
lossFKs, diceFlairFKs, diceT1cFKs = getLossForListOfRuns(fks)

#%%
sphericitysDTI, volumesDTI = getSphericity(patsDTI)
sphericitysFK, volumesFK = getSphericity(patsFK)

#%%
plt.scatter(patsDTI, sphericitysDTI, label="DTI")
plt.ylabel("Sphericity")
plt.xlabel("Patient")
#%% loss
plt.scatter(sphericitysDTI, 1 - np.array(lossDTIs), label="DTI")
plt.scatter(sphericitysFK, 1 - np.array(lossFKs), label="FK")
plt.legend()
plt.xlabel("Sphericity")
plt.ylabel("Volume Weighted Dice")

#%% weighted dice, loss
plt.figure(figsize=(10, 5))
plt.scatter(patsDTI, 1 - np.array(lossDTIs), label="DTI")
plt.scatter(patsFK, 1 - np.array(lossFKs), label="FK")
plt.xlabel("Patient")
plt.ylabel("Volume Weighted Dice")
plt.legend()
#%% plot dice flair
plt.scatter(patsDTI, diceFlairDTIs, label="DTI")
plt.scatter(patsFK, diceFlairFKs, label="FK")
plt.xlabel("Patient")
plt.ylabel("Dice Flair")
plt.legend()

#%% plot dice T1c
plt.scatter(patsDTI, diceT1cDTIs, label="DTI")
plt.scatter(patsFK, diceT1cFKs, label="FK")
plt.xlabel("Patient")
plt.ylabel("Dice T1c")
plt.legend()


#%%% plot runtime
runtimeDTIs = getRuntimeforListOfRuns(dtis)
runtimeFKs = getRuntimeforListOfRuns(fks)
plt.plot(patsDTI, np.array(runtimeDTIs) / 60, label="DTI")
plt.plot(patsFK, np.array(runtimeFKs) /60, label="FK")
plt.xlabel("Patient")
plt.ylabel("Runtime in hours")

plt.legend()

# %%
def getDiffs(pat1, res1, pat2, res2):
    dicesDiff, pats = [], []
    for pat in range(0, 3000):
        if str(pat) in pat1 and str(pat) in pat2:
            argwhere1 = np.argwhere(pat1 == str(pat))
            argwhere2 = np.argwhere(pat2 == str(pat))
            loss1 = res1[argwhere1[0][0]]
            loss2 = res2[argwhere2[0][0]]
            #print(pat)
            #print(loss1, loss2)
            diff = loss2 - loss1
            #print(diff)
            dicesDiff.append(diff)
            pats.append(str(pat))
    return dicesDiff, pats

weigtedDiceDiff, pats = getDiffs(patsFK, 1- np.array(lossFKs), patsDTI, 1-np.array(lossDTIs))
print("mean dice weighted diff", np.mean(weigtedDiceDiff), "+-", np.std(weigtedDiceDiff)/np.sqrt(len(weigtedDiceDiff)) , ",    std dice diff", np.std(weigtedDiceDiff))

flairDiceDiff, pats = getDiffs(patsFK, diceFlairFKs, patsDTI, diceFlairDTIs)
print("mean dice flair    diff", np.mean(flairDiceDiff), "+-", np.std(flairDiceDiff)/np.sqrt(len(flairDiceDiff)) , ",    std dice diff", np.std(flairDiceDiff))

t1cDiceDiff, pats = getDiffs(patsFK, diceT1cFKs, patsDTI, diceT1cDTIs)
print("mean dice T1c      diff", np.mean(t1cDiceDiff), "+-", np.std(t1cDiceDiff)/np.sqrt(len(t1cDiceDiff)) , ",    std dice diff", np.std(t1cDiceDiff))
#%%
plt.hist(weigtedDiceDiff, bins=10)
plt.title("Weighted Dice Diff")
plt.show()
#%% hist flair diff
plt.hist(flairDiceDiff, bins=10)
plt.title("Flair Dice Diff")
plt.show()
#%% hist t1c diff
plt.hist(t1cDiceDiff, bins=10)
plt.title("T1c Dice Diff")
plt.show()
#%%
sphericitys, volumes = getSphericity(pats)
plt.scatter(sphericitys, weigtedDiceDiff)
plt.xlabel("Sphericity")
plt.ylabel("Weighted Dice Diff")

# %% pairwise wilkoxon test
from scipy.stats import wilcoxon

print("weighted", wilcoxon(weigtedDiceDiff))
print("flair", wilcoxon(flairDiceDiff))
print("t1c", wilcoxon(t1cDiceDiff))



# %%

#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
import trimesh

#fkPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_28_testDTIFKOnly_init_larger_std_butterfly/"
fkPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_26_testDTI_fix_std/"# good run
#
# #"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/26_testDTI_fix_vol/"

#dtiPath ="/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_28_testDTIFK_init_larger_std_butterfly/" # gm homo

dtiPath ="/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_28_testDTI_no_homo_gm/" # full DTI



dtiPath ="/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_27_testDTI_init_larger_std/"
# 

# #/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_27_testDTI_init_larger_std/"#"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_26_testDTI_fix_vol/"# "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_23_testDTI_new/"#16_testDTI/"# "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_13_testDTI/"#

#evolutionary_sampling18_testFK_butterfly
originalTumorLocationPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_00016/preop/"

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

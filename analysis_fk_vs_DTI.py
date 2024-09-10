#%%
import numpy as np
import os
import matplotlib.pyplot as plt


fkPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_09_testFK/"

dtiPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_10_testDTI/" 

fkPathRuns = os.listdir(fkPath)
dtiPathRuns = os.listdir(dtiPath)


patsDTI, patsFK, fks, dtis = [], [], [], []
for pat in range(14, 300):
    patstring = f"BraTS2021_{str(pat).zfill(5)}"

    if os.path.exists(fkPath + patstring) or os.path.exists(dtiPath + patstring):
        if os.path.exists(dtiPath + patstring):
            dirs = os.listdir(dtiPath + patstring)
            for d in dirs:
                if "100_results.npy" in d or "101_results.npy" in d or "102_results.npy" in d:
                    dtis.append(np.load(f"{dtiPath}{patstring}/{d}", allow_pickle=True).item())
                    patsDTI.append(pat)               


        if os.path.exists(fkPath + patstring):
            dirs = os.listdir(fkPath + patstring)
            for d in dirs:
                if "100_results.npy" in d or "101_results.npy" in d or "102_results.npy" in d:
                    fks.append(np.load(f"{fkPath}{patstring}/{d}", allow_pickle=True).item())
                    patsFK.append(pat)
       
#%%
getLossForListOfRuns(runs):
    losses, diceFlairs, diceT1cs = [], [], []
    for run in runs:
        minLoss, opt, bestLossDir, diceFlair, diceT1c = getLossForListOFRuns(run)
        losses.append(minLoss)
        diceFlairs.append(diceFlair)
        diceT1cs.append(diceT1c)
        
    return minLoss, opt, bestLossDir, diceFlair, diceT1c

getRuntimeforListOfRuns(runs):
    runtimes = []
    for run in runs:
        runtimes.append(run["time_min"])
    return runtimes

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

    print("minLoss", minLoss)
    print("opt", opt)
    print("bestLossDir", bestLossDir)
    return minLoss, opt, bestLossDir, diceFlair, diceT1c

lossDTIs, lossFKs, diceFlairDTIs, diceT1cDTIs, diceFlairFKs, diceT1cFKs, pats = [], [], [], [], [], [], []


for pat in range(300):
    if pat in patsDTI and pat in patsFK:
        patWhere = patsDTI.index(pat)
        lossDTI,_,_, diceFlair, diceT1c = getLossOfRun(dtis[patWhere])
        lossDTIs.append(lossDTI)
        diceFlairDTIs.append(diceFlair)
        diceT1cDTIs.append(diceT1c)

        patWhere = patsFK.index(pat)
        lossFK,_,_, diceFlair, diceT1c = getLossOfRun(fks[patWhere])
        lossFKs.append(lossFK)
        diceFlairFKs.append(diceFlair)
        diceT1cFKs.append(diceT1c)

        pats.append(pat)
    print(pat)
pats = np.array(pats).astype(str)

#%% weighted dice, loss
plt.plot(pats, 1 - np.array(lossDTIs), label="DTI")
plt.plot(pats, 1 - np.array(lossFKs), label="FK")
plt.xlabel("Patient")
plt.ylabel("Volume Weighted Dice")
plt.legend()
#%% plot dice flair
plt.plot(pats, diceFlairDTIs, label="DTI")
plt.plot(pats, diceFlairFKs, label="FK")
plt.xlabel("Patient")
plt.ylabel("Dice Flair")
plt.legend()

#%% plot dice T1c
plt.plot(pats, diceT1cDTIs, label="DTI")
plt.plot(pats, diceT1cFKs, label="FK")
plt.xlabel("Patient")
plt.ylabel("Dice T1c")
plt.legend()

#%%
len(pats), len(fks), len(dtis)
#%%
path = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_09_testFK/BraTS2021_00014/sub-BraTS2021_00014_ses-preop_space-sri_gen_100_results.npy"

res = np.load(path, allow_pickle=True).item()

# %%

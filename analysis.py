#%%
"""
This file is for plotting of various results from the results.npy file
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


'''
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-02_05_02_gen_20/results.npy", allow_pickle=True).item()

res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-13_28_47_gen_20/results.npy", allow_pickle=True).item()

res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_13-23_12_33_gen_10/results.npy", allow_pickle=True).item()
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_13-22_19_32_gen_20/results.npy", allow_pickle=True).item()


res=np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_14-19_59_24_gen_50/results.npy", allow_pickle=True).item()

#good one 
res= np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_14-20_32_11_gen_200/results.npy", allow_pickle=True).item()

#largeTest full res
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_15-06_42_07_gen_1250/results.npy", allow_pickle=True).item()

#res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_15-10_52_25_gen_20/results.npy", allow_pickle=True).item()

# test with resolution increase in the very end 
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_17-00_16_39_gen_187/results.npy", allow_pickle=True).item()

# test with resolution increase in steps
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_17-01_13_20_gen_187/results.npy", allow_pickle=True).item()
'''
#Respond
#res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results/003/gen_112_results.npy", allow_pickle=True).item()

res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results_newSettings2/003/gen_112_results.npy", allow_pickle=True).item()

#res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results/121newSettings/gen_112_results.npy", allow_pickle=True).item()
#res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results/120newSettings/gen_112_results.npy", allow_pickle=True).item()

#res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results/051/gen_112_results.npy", allow_pickle=True).item()

res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results/016OldLoss/gen_112_results.npy", allow_pickle=True).item()

res = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results_9NewStandard/016/gen_25_results.npy", allow_pickle=True).item()


# %%
res.keys()
res["variableParameters"]
#%%
np.array(res["Cs"])[-1]


#%%
lossDir = res["lossDir"]
minLoss = 1
for i in range(len(lossDir)):
    for j in range(len(lossDir[i])):
        if lossDir[i][j]["lossTotal"] < minLoss:
            minLoss = lossDir[i][j]["lossTotal"]
            opt = lossDir[i][j]["allParams"]
            bestLossDir = lossDir[i][j]

print("minLoss", minLoss)
print("opt", opt)
print("bestLossDir", bestLossDir)

print("----------------------------")

print("best Total loss ", bestLossDir["lossTotal"])
#print("best Pet corr   ", 1-bestLossDir["lossPet"])
print("best T1c dice   ", 1-bestLossDir["lossT1c"])
print("best Flair dice ", 1-bestLossDir["lossFlair"])

#%%

lossT1c, lossFlair, times, xs, resfactor, totalLoss = [], [], [], [], [], []
for i in range(len(res["lossDir"])):
    lossT1c_, lossFlair_, times_, xs_, resfactor_, totalLoss_ = [], [], [], [], [], []
    for j in range(len(res["lossDir"][i])):
        if not res["lossDir"][i][j]["lossTotal"] <=1:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        if not res["lossDir"][i][j]["lossTotal"] >=0:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        if not res["lossDir"][i][j]["lossTotal"] == res["lossDir"][i][j]["lossTotal"]:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        lossT1c_.append(res["lossDir"][i][j]["lossT1c"])
        lossFlair_.append(res["lossDir"][i][j]["lossFlair"])
        times_.append(res["lossDir"][i][j]["time"])
        xs_.append(res["lossDir"][i][j]["allParams"])
        try:
            resfactor_.append(res["lossDir"][i][j]["resolution_factor"])
        except:
            resfactor_.append(1)
        try:
            totalLoss_.append(res["lossDir"][i][j]["lossTotal"])
        except:
            totalLoss_.append(1)
    lossT1c.append(lossT1c_)
    lossFlair.append(lossFlair_)
    times.append(times_)
    xs.append(xs_)
    resfactor.append(resfactor_)
    totalLoss.append(totalLoss_)

times = np.array(times )/60
xs = np.array(xs)
lossCombined = 0.5 * np.array(lossFlair) + 0.5 * np.array(lossT1c)

#%%
plt.figure(figsize=(12, 7))	


plt.errorbar(res["nsamples"], np.mean(lossT1c, axis=1), yerr=np.std(lossT1c, axis=1),   label="lossT1c")
plt.errorbar(res["nsamples"], np.mean(lossFlair, axis=1), yerr=np.std(lossFlair, axis=1),   label="lossFlair")
plt.errorbar(res["nsamples"], np.mean(lossCombined, axis=1) , yerr=np.std(lossCombined, axis=1),   label="combinedLoss")
plt.errorbar(res["nsamples"], np.mean(totalLoss, axis=1) , yerr=np.std(totalLoss, axis=1),   label="totalLoss")
plt.ylabel("loss")
plt.xlabel("number of samples")
plt.legend()
#%%
plt.figure(figsize=(12, 7))	

plt.plot(res["nsamples"], res["sigmas"])
plt.ylabel("sigma")
plt.xlabel("number of samples")

#%%
def plotValues(values, yLab, title):
    plt.figure(figsize=(12, 7))	
    for i in range(len((values))):
        plt.scatter([res["nsamples"][i]]*len(values[i]), values[i], color="tab:blue", marker=".")
    plt.ylabel(yLab)
    plt.xlabel("Samples")
    plt.title(yLab + " ---  " + title)
    plt.show()

title = "Samples: " + np.max(res["nsamples"]).astype(str) + " - cumulative time: " + str(np.round(np.sum(times), 2)) + " - parallel time: " + str(np.round(res["time_min"], 1)) + "min"
plotValues(times, "time [min]", title)
#%%
plotValues(resfactor, "resolution_factor", title)

#%%

#%%
for i in range(10):
    plotValues(xs[:,:,i], res["variableParameters"][i], title)
#%%
plt.figure(figsize=(12, 7))	
plt.title(title)
plt.errorbar(res["nsamples"], np.mean(times, axis=1), yerr=np.std(times, axis=1), fmt='o', color='black', label="time")
plt.ylabel("time [min]")
plt.xlabel("nsamples")
plt.legend()

# %%
res.keys()

# %%
params = np.array(res["xs0s"])
plt.title("thresholds")
plt.plot(res["nsamples"], params.T[6], label="Edema")
plt.plot(res["nsamples"], params.T[5], label="Enhancing")
plt.legend()
plt.xlabel("nsamples")
plt.show()

# %%
plt.title("rho - D")
plt.plot(res["nsamples"], params.T[3], label="rho")
plt.plot(res["nsamples"],params.T[4], label="D")
plt.xlabel("nsamples")
plt.legend()

# %%
x = params.T[0]   
y = params.T[1]
z = params.T[2]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming you have x, y, and z data
# Create a colormap based on the z values
colors = plt.cm.jet(np.linspace(0, 1, len(z)))

# Plot each segment with a different color
for i in range(1, len(x)):
    ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=colors[i])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Origin")
plt.show()
# %% only debug
import plotly.graph_objects as go

# Assuming you have x, y, and z data
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines',
    line=dict(
        color=np.arange(len(x)),  # set color to an array/list of desired values
        colorscale='Viridis',     # choose a colorscale
        width=2
    )
)])

fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                  width=700,
                  margin=dict(r=20, l=10, b=10, t=10))

fig.show()


# %%
from TumorGrowthToolkit.FK import Solver as fwdSolver
parameters = {
            'Dw': x[4],         # Diffusion coefficient for white matter
            'rho': x[3],        # Proliferation rate
            'RatioDw_Dg': 10,   # Ratio of diffusion coefficients in white and grey matter
            'gm': self.wm,      # Grey matter data
            'wm': self.gm,      # White matter data
            'NxT1_pct': x[0],   # initial focal position (in percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor':self.settings["resolution_factor"]
        }
print("run: ", x)
fwdRes =  fwdSolver(parameters)["final_state"]
fwdSolver.run()
# %%

#%%
import numpy as np
import matplotlib.pyplot as plt
from FK_2c_cmaes import writeNii

base_path = '/Users/michal/Documents/cmaesForPythonFWD/results/2023_12_21-13_38_16_gen_10/'
#%%
# Load results
res = np.load(f"{base_path}/results.npy", allow_pickle=True).item()

#%%
# Find the minimum total loss and optimal parameters
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

#%%
# Initialize arrays for losses and parameters
lossEdema, lossEnhancing, lossNecrotic, times, xs, resfactor = [], [], [], [], [], []
for i in range(len(res["lossDir"])):
    lossEdema_, lossEnhancing_, lossNecrotic_, times_, xs_, resfactor_ = [], [], [], [], [], []
    for j in range(len(res["lossDir"][i])):
        lossEdema_.append(res["lossDir"][i][j]["lossEdema"])
        lossEnhancing_.append(res["lossDir"][i][j]["lossEnhancing"])
        lossNecrotic_.append(res["lossDir"][i][j]["lossNecrotic"])
        times_.append(res["lossDir"][i][j]["time"])
        xs_.append(res["lossDir"][i][j]["allParams"])
        resfactor_.append(res["lossDir"][i][j]["resolution_factor"])
    lossEdema.append(lossEdema_)
    lossEnhancing.append(lossEnhancing_)
    lossNecrotic.append(lossNecrotic_)
    times.append(times_)
    xs.append(xs_)
    resfactor.append(resfactor_)

times = np.array(times )/60
xs = np.array(xs)

#%%
# Function to plot values
def plotValues(values, yLab, title):
    plt.figure(figsize=(12, 7))
    for i in range(len(values)):
        plt.scatter([res["nsamples"][i]]*len(values[i]), values[i], color="tab:blue", marker=".")
    plt.ylabel(yLab)
    plt.xlabel("Generation")
    plt.title(yLab + " ---  " + title)
    plt.show()

title = "Samples: " + np.max(res["nsamples"]).astype(str) + " - cumulative time: " + str(np.round(np.sum(times), 2)) + " - parallel time: " + str(np.round(res["time_min"], 1)) + "min"

# Plotting time and resolution factor
plotValues(times, "time [min]", title)
plotValues(resfactor, "resolution_factor", title)

# Plotting new loss components
plotValues(lossEdema, "lossEdema", title)
plotValues(lossEnhancing, "lossEnhancing", title)
plotValues(lossNecrotic, "lossNecrotic", title)

# Plotting parameters (assuming they are at specific indices in 'xs')
# Update the indices as per your 'xs' array structure
plotValues(xs[:,:,3], "Parameter1", title) # Replace 'Parameter1' with actual parameter name
plotValues(xs[:,:,4], "Parameter2", title) # Replace 'Parameter2' with actual parameter name
# Add more plots for other parameters as needed

#%%
# Plot for time error bar
plt.figure(figsize=(12, 7))
plt.title(title)
plt.errorbar(res["nsamples"], np.mean(times, axis=1), yerr=np.std(times, axis=1), fmt='o', color='black', label="time")
plt.ylabel("time [min]")
plt.xlabel("nsamples")
plt.legend()

#%%
# Plot for combined loss
plt.figure(figsize=(12, 7))
combinedLoss = (np.array(lossEdema) + np.array(lossEnhancing) + np.array(lossNecrotic) )/3
plt.errorbar(res["nsamples"], np.mean(combinedLoss, axis=1), yerr=np.std(combinedLoss, axis=1), label="combinedLoss")
plt.ylabel("loss")
plt.xlabel("number of samples")
plt.legend()

# Add any additional plotting or analysis as required

# %%
from TumorGrowthToolkit.FK_2c import Solver
import nibabel as nib
# Get probability images for Grey and White matter
GM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/gm_data.nii.gz").get_fdata()
WM =  nib.load("/Users/michal/Documents/TumorGrowthToolkit/synthetic_gens/synthetic_runs1T_FK_2c/synthetic1T_run0/wm_data.nii.gz").get_fdata()
params = np.array(res["xs0s"])
x = params[-1]
parameters = {
    'Dw': x[4],
    'rho': x[3],
    'lambda_np': x[5],
    'sigma_np': x[6],
    'D_s': x[7],
    'lambda_s': x[8],
    'gm': GM,
    'wm': WM,
    'NxT1_pct': x[0],
    'NyT1_pct': x[1],
    'NzT1_pct': x[2],
    'resolution_factor': 0.5
}

# Run the FK_solver and plot the results
fk_solver = Solver(parameters)
result = fk_solver.solve()
writeNii(result['final_state'], base_path)
# %%

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-02_05_02_gen_20/results.npy", allow_pickle=True).item()

res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-13_28_47_gen_20/results.npy", allow_pickle=True).item()

res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_13-23_12_33_gen_10/results.npy", allow_pickle=True).item()
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_13-22_19_32_gen_20/results.npy", allow_pickle=True).item()


res=np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_14-19_59_24_gen_50/results.npy", allow_pickle=True).item()
res= np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/resultsP001/2023_12_14-20_32_11_gen_200/results.npy", allow_pickle=True).item()

# %%
res.keys()

#%%
res["lossDir"][30]
#%%

lossPet, lossT1c, lossFlair, times = [], [], [], []
for i in range(len(res["lossDir"])):
    lossPet_, lossT1c_, lossFlair_, times_ = [], [], [], []
    for j in range(len(res["lossDir"][i])):
        if not res["lossDir"][i][j]["lossTotal"] <=1:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        if not res["lossDir"][i][j]["lossTotal"] >=0:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        if not res["lossDir"][i][j]["lossTotal"] == res["lossDir"][i][j]["lossTotal"]:
            print("error", i, j, res["lossDir"][i][j]["lossTotal"])
        lossPet_.append(res["lossDir"][i][j]["lossPet"])
        lossT1c_.append(res["lossDir"][i][j]["lossT1c"])
        lossFlair_.append(res["lossDir"][i][j]["lossFlair"])
        times_.append(res["lossDir"][i][j]["time"])
    lossPet.append(lossPet_)
    lossT1c.append(lossT1c_)
    lossFlair.append(lossFlair_)
    times.append(times_)

times = np.array(times )/60

#%%
plt.figure(figsize=(12, 7))	
def plotValues(values, yLab, title):
    for i in range(len((values))):
        plt.scatter([res["nsamples"][i]]*len(values[i]), values[i], color="tab:blue", marker=".")
    plt.ylabel(yLab)
    plt.xlabel("Generation")
    plt.title(title)
    plt.show()

title = "Samples: " + np.max(res["nsamples"]).astype(str) + " - cumulative time: " + str(np.round(np.sum(times), 2)) + " - parallel time: " + str(np.round(res["time_min"], 1)) + "min"
plotValues(times, "time [min]", title)

#plotValues(lossPet, "lossPet")
#plotValues(lossT1c, "lossT1c")
#plotValues(lossFlair, "lossFlair")

#%%
plt.figure(figsize=(12, 7))	
plt.title(title)
plt.errorbar(res["nsamples"], np.mean(times, axis=1), yerr=np.std(times, axis=1), fmt='o', color='black', label="time")
plt.ylabel("time [min]")
plt.xlabel("nsamples")
plt.legend()

#%%
plt.figure(figsize=(12, 7))	

plt.errorbar(res["nsamples"], np.mean(lossPet, axis=1), yerr=np.std(lossPet, axis=1), label="lossPet")
plt.errorbar(res["nsamples"], np.mean(lossT1c, axis=1), yerr=np.std(lossT1c, axis=1),   label="lossT1c")
plt.errorbar(res["nsamples"], np.mean(lossFlair, axis=1), yerr=np.std(lossFlair, axis=1),   label="lossFlair")
plt.ylabel("loss")
plt.xlabel("number of samples")
plt.legend()
#%%
plt.figure(figsize=(12, 7))	

plt.plot(res["nsamples"], res["sigmas"])
plt.ylabel("sigma")
plt.xlabel("number of samples")

# %%
res.keys()

# %%
params = np.array(res["xs0s"])
plt.title("thresholds")
plt.plot(res["nsamples"], params.T[6], label="T1c")
plt.plot(res["nsamples"], params.T[5], label="Flair")
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

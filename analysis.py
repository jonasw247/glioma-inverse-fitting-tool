#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%
res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-02_05_02_gen_20/results.npy", allow_pickle=True).item()

res = np.load("/home/jonas/workspace/programs/cmaesForPhythonFWD/results/2023_12_07-13_28_47_gen_20/results.npy", allow_pickle=True).item()
# %%
res["time_min"]
#%%
plt.plot()
#%%

# %%
lossPet = [d[0]["lossPet"] for d in res["lossDir"]]
lossT1c = [d[0]["lossT1c"] for d in res["lossDir"]]
lossFlair = [d[0]["lossFlair"] for d in res["lossDir"]]

# %%
plt.plot(lossPet)
#%%
plt.plot(lossT1c, label="lossT1c")
plt.plot(lossFlair, label="lossFlair")
plt.legend()

# %%
plt.plot(res["y0s"])

# %%
res.keys()

# %%
lala = np.array(res["xs0s"])
# %%
plt.title("thresholds")
plt.plot(lala.T[6])
plt.plot(lala.T[5])

# %%
x = lala.T[0]   
y = lala.T[1]
z = lala.T[2]

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
# %%
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
plt.title("rho - D")
plt.plot(res["nsamples"], lala.T[3], label="rho")
plt.plot(res["nsamples"],lala.T[4], label="D")
plt.xlabel("nsamples")
plt.legend()

# %%
plt.plot(lala.T[3], lala.T[4])

# %%

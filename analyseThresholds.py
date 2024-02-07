#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

paths = ["/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results_newSettings2/", "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_results_NoPet/", "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_results_withoutPet/"]

pathPart2 = "/gen_112_results.npy"


# %%
t1cParams = []
flairParams = []
times = []
for i in range(0,200):
    for path in paths:
        try:
            
            strPat = ("0000" + str(i))[-3:]
            res = np.load(path + strPat + pathPart2, allow_pickle=True).item() 
            t1cParams.append(res["opt_params"][5])
            flairParams.append(res["opt_params"][6])
            times.append(res["time_min"])
        except:
            continue

# %%
res.keys()
# %%
plt.title("T1c")
plt.hist(t1cParams, bins=20)
plt.show()
plt.title("flair")

plt.hist(flairParams, bins=20)
plt.show()
plt.title("time")

plt.hist(times, bins=20)
plt.show()
# %%
print("time mean ", np.mean(times))
print("time std ", np.std(times))
# %%

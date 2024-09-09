#%%
import numpy as np

fkPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_09_testFK/"

dtiPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_09_testDTI/" 


path = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/cma-es_results/cma-es_results_09_testFK/BraTS2021_00014/sub-BraTS2021_00014_ses-preop_space-sri_gen_100_results.npy"

res = np.load(path, allow_pickle=True).item()

# %%

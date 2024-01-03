#%%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
#%% Load synthetic data of necrotic and proliferative tumor
proliferativeGroundTruth = nib.load("exampleData/P.nii.gz").get_fdata()
necroticGroundTruth = nib.load("exampleData/N.nii.gz").get_fdata()

plt.title("Synthetic Ground Truth - Necrotic (Red) and Proliferative (Blue) Tumor")
zSlice = 75
plt.imshow(proliferativeGroundTruth[:, :, zSlice],  cmap="Blues", alpha=proliferativeGroundTruth[:, :, zSlice])
plt.imshow(necroticGroundTruth[:, :, zSlice],  cmap="Reds", alpha=necroticGroundTruth[:, :, zSlice])

#%% Binary segmentation is need for the inverse model
thresholdEdema = 0.25
thresholdEnahncing = 0.4
segmentation = 0* proliferativeGroundTruth

# edema part
segmentation[proliferativeGroundTruth + necroticGroundTruth > thresholdEdema] = 3

# enhancing part 
segmentation[proliferativeGroundTruth + necroticGroundTruth > thresholdEnahncing] = 4

# necrotic part
segmentation[necroticGroundTruth > proliferativeGroundTruth] = 1

plt.imshow(segmentation[:, :, zSlice], alpha=0.99 * (segmentation[:, :, zSlice] >0), interpolation="none")

nib.save(nib.Nifti1Image(segmentation, np.eye(4)), "exampleData/tumor_seg.nii.gz")



#%%
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy import ndimage
for i in range(1,71):
    try:
        print(i)
        strPat = ("00000" + str(i))[-3:]
        plt.figure()
        plt.title(strPat)
        
        segmentation = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm" +strPat + "/preop/sub-tgm" +strPat + "_ses-preop_space-sri_seg.nii.gz").get_fdata() >0

        t1 = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm" +strPat + "/preop/sub-tgm" +strPat + "_ses-preop_space-sri_t1c.nii.gz").get_fdata()

        com = ndimage.measurements.center_of_mass(segmentation)
        plt.imshow(t1[:, int(com[1]), :],  cmap="Greys")
        values = segmentation[:,int(com[1]), : ]
        plt.imshow(values,  cmap="Reds", alpha=values.astype(float) * 0.8)
        plt.show()

    except:
        print("error")
        pass
# %%
for i in range(1,180):
    try:
        print(i)
        strPat = ("00000" + str(i))[-3:]
        plt.figure()
        plt.title(strPat)
        
        segmentation = nib.load("/mnt/8tb_slot8/jonas/datasets/ReSPOND/respond/respond_tum_" +strPat + "/d0/sub-respond_tum_" +strPat + "_ses-d0_space-sri_seg.nii.gz").get_fdata() >0

        t1 = nib.load("/mnt/8tb_slot8/jonas/datasets/ReSPOND/respond/respond_tum_" +strPat + "/d0/sub-respond_tum_" +strPat + "_ses-d0_space-sri_t1c.nii.gz").get_fdata() 

        com = ndimage.measurements.center_of_mass(segmentation)
        plt.imshow(t1[:, int(com[1]), :],  cmap="Greys")
        values = segmentation[:,int(com[1]), : ]
        plt.imshow(values,  cmap="Reds", alpha=values.astype(float) * 0.8)
        plt.show()

    except:
        print("error")
        pass
# %%

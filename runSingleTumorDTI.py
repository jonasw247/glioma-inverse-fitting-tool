#%%
import TumorGrowthToolkit.FK_DTI.tools as toolsDTI
from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#this set of parameters is not working but takes forever to run	
global_parameters_dir = {'Dw': 1.0, 'rho': 0.01, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 1, 'NxT1_pct': 0.41032485842150356, 'NyT1_pct': 0.682107039709935, 'NzT1_pct': 0.6776180829555788, 'resolution_factor': 0.5, 'stopping_volume': 113813.51104931717, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True} #resulution was 0.5 'NxT1_pct': 0.21032485842150356,

global_parameters_dir =  {'Dw': 1.0, 'rho': 0.0551608257797072, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 1, 'NxT1_pct': 0.6344179953484931, 'NyT1_pct': 0.4026485726596266, 'NzT1_pct': 0.6759434325906947, 'resolution_factor': 1.0, 'stopping_volume': 100532.92547047537, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True}

global_parameters_dit =  {'Dw': 1.0, 'rho': 0.738079774347954, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': None, 'NxT1_pct': 0.5470669460722168, 'NyT1_pct': 0.48367631804786604, 'NzT1_pct': 0.1538186519991361, 'resolution_factor': 0.5, 'stopping_volume': 48049.131682033956, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True, 'difffusionEllipsoidScaling': None}


{'Dw': 59.551555037015206, 'rho': 0.5, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 1, 'NxT1_pct': 0.35520135349515986, 'NyT1_pct': 0.7184248473917638, 'NzT1_pct': 0.3682253046477162, 'resolution_factor': 0.5, 'stopping_volume': 165057.71113718097, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True}


{'Dw': 80.54207890889282, 'rho': 0.5, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 0.5293293089737141, 'NxT1_pct': 0.3035039321610199, 'NyT1_pct': 0.6903178541615499, 'NzT1_pct': 0.5131769722294754, 'resolution_factor': 0.5, 'stopping_volume': 99370.9174813795, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True}

global_parameters_dit ={'Dw': 6.677491946210937, 'rho': 0.5, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 0.9878505948960314, 'NxT1_pct': 0.6620214456801183, 'NyT1_pct': 0.43053226140594536, 'NzT1_pct': 0.23789204632233127, 'resolution_factor': 0.5, 'stopping_volume': 43775.12805199479, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True}

global_parameters_dir = {'Dw': 43.23043942916391, 'rho': 0.5, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 0.002, 'NxT1_pct': 0.6292156400394847, 'NyT1_pct': 0.48049674542355136, 'NzT1_pct': 0.1978914068624885, 'resolution_factor': 0.5, 'stopping_volume': 33834.691961566048, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True, 'use_homogen_gm': True, 'RatioDw_Dg': 10.0}

patientID = 14

patString = ("000000" + str(patientID))[-5:]
try:
    segmPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c_smoothed_and_masked/BraTS2021_{patString}/preop/sub-BraTS2021_{patString}_ses-preop_space-sri_seg.nii.gz"
    dtiPath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{patString}/transformed_reoriented_tensor.nii.gz"
    tissuePath = f"/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas/BraTS2021_{patString}/transformed_tissue.nii.gz"

    segm = nib.load(segmPath)
    segmentation = segm.get_fdata()
    affine = segm.affine

    brainTissue = nib.load(tissuePath).get_fdata()
    diffusionTensorsLower = nib.load(dtiPath).get_fdata()[:, :, :, 0, :]
    diffusionTensors = toolsDTI.get_tensor_from_lower6(diffusionTensorsLower)

    print("found data for patient", patientID)
except Exception as e:
    print(f"patient {patientID} not found: {e}")
    

CSFMask = brainTissue == 1
CSFMask[segmentation > 0] = 0
diffusionTensors[CSFMask] = 0
brainmask = brainTissue > 0

edema = np.logical_or(segmentation == 3, segmentation == 2)
necrotic = segmentation == 1
enhancing = segmentation == 4



global_parameters_dir["diffusionTensors"] = diffusionTensors

wm = brainTissue == 3
gm = brainTissue == 2

#exclude CSF
CSFMask = brainTissue == 1
# include tumor segmentation region,
mask = CSFMask.copy()
mask[segmentation > 0] = 0
diffusionTensors[mask] = 0

#exclude CSF 
wm[mask] = 0
gm[mask] = 0
gm[np.logical_and(CSFMask, segmentation>0)] = True

global_parameters_dir["gm"] = gm * 1.0
global_parameters_dir["wm"] = wm *1.0
fK_DTI_Solver = FK_DTI_Solver(global_parameters_dir)

#plot start
brainmask = np.sum(np.sum(diffusionTensors, axis=-1),axis=-1) 
tissue = brainmask*0.2 +edema *0.3 + enhancing * 0.6 + necrotic * 0.8
x = int(tissue.shape[0]*global_parameters_dir["NxT1_pct"])
y = int(tissue.shape[1]*global_parameters_dir["NyT1_pct"])
z = int(tissue.shape[2]*global_parameters_dir["NzT1_pct"])

plt.imshow(brainmask[:,:,z])
plt.scatter(y,x, c='r')
plt.title("Tumor Origin")
plt.show()
result = fK_DTI_Solver.solve(doPlot=True)
print(result)

#%%# %%
#plt.imshow(brainTissue[:,:,z], cmap="gray")
plt.imshow(tissue[:,:,z], cmap='gray', alpha=0.5)
tumor = result["final_state"][:,:,z]
plt.imshow(tumor, cmap='Reds' , alpha= tumor)
# %%

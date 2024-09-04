#%%
import TumorGrowthToolkit.FK_DTI.tools as toolsDTI
from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#this set of parameters is not working but takes forever to run	
global_parameters_dir = {'Dw': 1.0, 'rho': 0.01, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': 1, 'NxT1_pct': 0.41032485842150356, 'NyT1_pct': 0.682107039709935, 'NzT1_pct': 0.6776180829555788, 'resolution_factor': 0.5, 'stopping_volume': 113813.51104931717, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True} #resulution was 0.5 'NxT1_pct': 0.21032485842150356,



def process_patient(patientID):

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
        return

    CSFMask = brainTissue == 1
    CSFMask[segmentation > 0] = 0
    diffusionTensors[CSFMask] = 0
    brainmask = brainTissue > 0

    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4

    if np.sum(edema) + np.sum(necrotic) + np.sum(enhancing) < 25 or np.sum(edema) < 25:
        print("Too small tumor for patient", patientID)
        return

    global_parameters_dir["diffusionTensors"] = diffusionTensors
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


if __name__ == '__main__':
    process_patient(115)
# %%

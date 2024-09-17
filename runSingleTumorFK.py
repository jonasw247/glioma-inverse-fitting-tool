#%%
import TumorGrowthToolkit.FK_DTI.tools as toolsDTI
from TumorGrowthToolkit.FK import Solver as Fk_Solver
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

global_parameters_dir =  {'Dw': 1.0, 'rho': 0.738079774347954, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': None, 'NxT1_pct': 0.670669460722168, 'NyT1_pct': 0.42367631804786604, 'NzT1_pct': 0.0003538186519991361, 'resolution_factor': 0.5, 'stopping_volume': 48049.131682033956, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True, 'difffusionEllipsoidScaling': None}

global_parameters_dir = {'Dw': 1.0, 'rho': 10.0, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': None, 'NxT1_pct': 0.5828975569055244, 'NyT1_pct': 0.4397344617309852, 'NzT1_pct': 0.16969136432951953, 'resolution_factor': 0.5, 'stopping_volume': 55355.93922186011, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True, 'difffusionEllipsoidScaling': None}

global_parameters_dir = {'Dw': 1.0, 'rho': 5, 'diffusionEllipsoidScaling': 1, 'diffusionTensorExponent': None, 'NxT1_pct': 0.6388018763342702, 'NyT1_pct': 0.7025559698303504, 'NzT1_pct': 0.3516943091717649, 'resolution_factor': 0.5, 'stopping_volume': 65086.167678080834, 'stopping_time': 10000000, 'init_scale': 1.0, 'verbose': True, 'difffusionEllipsoidScaling': None}


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

    
    brainmask = brainTissue > 0

    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4


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

    if np.sum(edema) + np.sum(necrotic) + np.sum(enhancing) < 25 or np.sum(edema) < 25:
        print("Too small tumor for patient", patientID)
        return

    global_parameters_dir["diffusionTensors"] = diffusionTensors
    global_parameters_dir["wm"] = wm
    global_parameters_dir["gm"] = gm
    solver = Fk_Solver(global_parameters_dir)

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
    result = solver.solve()
    print(result)

    #plot tumor
    plt.imshow(result["final_state"][:,:,z])


if __name__ == '__main__':
    process_patient(115)
# %%

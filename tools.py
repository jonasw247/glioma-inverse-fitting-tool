import nibabel as nib
import numpy as np

def writeNii(array, path = "", affine = np.eye(4)):
    if path == "":
        path = "%dx%dx%dle.nii.gz" % np.shape(array)
    nibImg = nib.Nifti1Image(array, affine)
    nib.save(nibImg, path)
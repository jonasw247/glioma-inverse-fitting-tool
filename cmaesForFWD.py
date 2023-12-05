
import cmaes
import numpy as np
import nibabel as nib
import time
import nibabel as nib
from forwardFK_FDM.solver import solver as fwdSolver

        
def dice(a, b):
    boolA, boolB = a > 0, b > 0 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0

    return 2 * np.sum( np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

def writeNii(array, path = "", affine = np.eye(4)):
    if path == "":
        path = "%dx%dx%dle.nii.gz" % np.shape(array)
    nibImg = nib.Nifti1Image(array, affine)
    nib.save(nibImg, path)


class CmaesSolver():
    def __init__(self,  settings, wm, gm, flair, t1c, pet):
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.flair = flair
        self.t1c = t1c
        self.pet = pet


    def lossfunction(self, tumor):

        lambdaFlair = 0.333
        lambdaT1c = 0.333
        lambdaPET = 0.333

        thresholdT1c = 0.675
        thresholdFlair = 0.25

        lossPet = np.mean(np.abs(tumor - self.pet))
        lossFlair = 1 - dice(tumor > thresholdFlair, self.flair)
        lossT1c = 1 - dice(tumor > thresholdT1c, self.t1c)
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c + lambdaPET * lossPet

        return loss, {"lossFlair":lossFlair ,"lossT1c": lossT1c, "lossPet":lossPet}



    def forward(self, x):

        parameters = {
            'Dw': x[4],         # Diffusion coefficient for white matter
            'rho': x[3],        # Proliferation rate
            'RatioDw_Dg': 10,   # Ratio of diffusion coefficients in white and grey matter
            'gm': self.wm,      # Grey matter data
            'wm': self.gm,      # White matter data
            'NxT1_pct': x[0],   # initial focal position (in percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2]
        }
        print("run: ", x)
        return fwdSolver(parameters)["final_state"]


    def getLoss(self, x):

        tumor = self.forward(x)
        
        loss, lossDir = self.lossfunction(tumor)
        
        print("loss: ", loss, "lossDir: ", lossDir, "x: ", x)

        return loss, lossDir

    def run(self):
        start = time.time()
        
        trace = cmaes.cmaes(self.getLoss, ( self.settings["NxT1_pct0"], self.settings["NyT1_pct0"], self.settings["NzT1_pct0"], self.settings["dw0"], self.settings["rho0"]), self.settings["sigma0"], self.settings["generations"], workers=self.settings["workers"], trace=True, parameterRange= self.settings["parameterRanges"])

        #trace = np.array(trace)
        nsamples, y0s, xs0s, sigmas, Cs, pss, pcs, Cmus, C1s, xmeans, lossDir = [], [], [], [], [], [], [], [], [], [], []
        for element in trace:
            nsamples.append(element[0])
            y0s.append(element[1])
            xs0s.append(element[2])
            sigmas.append(element[3])
            Cs.append(element[4])
            pss.append(element[5])
            pcs.append(element[6])
            Cmus.append(element[7])
            C1s.append(element[8])
            xmeans.append(element[9])
            lossDir.append(element[10])

        opt = xmeans[-1]

        tumor = self.forward(opt)
        end = time.time()

        resultDict = {}

        resultDict["nsamples"] = nsamples
        resultDict["y0s"] = y0s
        resultDict["xs0s"] = xs0s
        resultDict["sigmas"] = sigmas
        resultDict["Cs"] = Cs
        resultDict["pss"] = pss
        resultDict["pcs"] = pcs
        resultDict["Cmus"] = Cmus
        resultDict["C1s"] = C1s
        resultDict["xmeans"] = xmeans
        resultDict["lossDir"] = lossDir

        resultDict["opt_params"] = opt
        resultDict["time_min"] = (end - start) / 60
        
        return tumor, resultDict

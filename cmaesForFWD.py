
import cmaes
import numpy as np
import nibabel as nib
import time
import nibabel as nib
from forwardFK_FDM.solver import solver as fwdSolver
import time

        
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
    def __init__(self,  settings, wm, gm, flair, enhancing, pet, necrotic):
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.edema = flair
        self.enhancing = enhancing
        self.pet = pet
        self.necrotic = necrotic


    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):

        lambdaFlair = 0.333
        lambdaT1c = 0.333
        lambdaPET = 0.333

        petInsideTumorRegion = self.pet * np.logical_or(self.edema, self.enhancing)
        lossPet = 1 - np.corrcoef(tumor.copy().flatten(), petInsideTumorRegion.copy().flatten() )[0,1]
        
        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c	)
        lossFlair = 1 - dice(proposedEdema, self.edema)
        lossT1c = 1 - dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
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
            'NzT1_pct': x[2],
            'resolution_factor':self.settings["resolution_factor"]
        }
        print("run: ", x)
        return fwdSolver(parameters)["final_state"]


    def getLoss(self, x):

        start_time = time.time()
        tumor = self.forward(x[:-2])
        
        thresholdT1c = x[-2]	
        thresholdFlair = x[-1]
        loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
        end_time = time.time()

        lossDir["time"] = end_time - start_time
        
        print("loss: ", loss, "lossDir: ", lossDir, "x: ", x)

        return loss, lossDir

    def run(self):
        start = time.time()
        
        initValues = (self.settings["NxT1_pct0"], self.settings["NyT1_pct0"], self.settings["NzT1_pct0"], self.settings["dw0"], self.settings["rho0"], self.settings["thresholdT1c"], self.settings["thresholdFlair"])

        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], self.settings["generations"], workers=self.settings["workers"], trace=True, parameterRange= self.settings["parameterRanges"])

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

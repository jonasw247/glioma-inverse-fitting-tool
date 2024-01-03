from TumorGrowthToolkit.FK import Solver as fwdSolver
import cmaes
import numpy as np
import nibabel as nib
import time
        
def dice(a, b):
    boolA, boolB = a > 0, b > 0 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0

    return 2 * np.sum( np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

class CmaesSolver():
    def __init__(self,  settings, wm, gm, edema, enhancing, pet, necrotic):
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.edema = edema
        self.enhancing = enhancing
        self.pet = pet
        self.necrotic = necrotic


    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):

        lambdaFlair = 0.333
        lambdaT1c = 0.333
        lambdaPET = 0.333

        petInsideTumorRegion = self.pet * np.logical_or(self.edema, self.enhancing)
        if np.sum(petInsideTumorRegion) == 0:
            lossPet = 1
        else:   
            lossPet = 1 - np.corrcoef(tumor.copy().flatten(), petInsideTumorRegion.copy().flatten() )[0,1]
        
        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c	)
        lossFlair = 1 - dice(proposedEdema, self.edema)
        lossT1c = 1 - dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c + lambdaPET * lossPet

        #catch none values
        if not loss<=1:
            loss = 1

        return loss, {"lossFlair":lossFlair ,"lossT1c": lossT1c, "lossPet":lossPet, "lossTotal":loss}


    def forward(self, x, resolution_factor =1.0):

        parameters = {
            'Dw': x[4],         # Diffusion coefficient for white matter
            'rho': x[3],        # Proliferation rate
            'RatioDw_Dg': 10,   # Ratio of diffusion coefficients in white and grey matter
            'gm': self.wm,      # Grey matter data
            'wm': self.gm,      # White matter data
            'NxT1_pct': x[0],   # initial focal position (in percentages)
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor':resolution_factor
        }
        print("run: ", x)
        solver = fwdSolver(parameters)
        return solver.solve()["final_state"]


    def getLoss(self, x, gen):

        start_time = time.time()

        #check if resolution factor is float or dict
        if isinstance(self.settings["resolution_factor"], dict):

            for relativeGen, resFactor in self.settings["resolution_factor"].items():
                if  gen /self.settings["generations"] >=  relativeGen :
                    resolution_factor = resFactor
        
        elif isinstance(self.settings["resolution_factor"], float):
            resolution_factor = self.settings["resolution_factor"]
        else:
            raise ValueError("resolution_factor has to be float or dict")
        
        tumor = self.forward(x[:-2], resolution_factor)
        
        thresholdT1c = x[-2]	
        thresholdFlair = x[-1]
        loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
        end_time = time.time()

        lossDir["time"] = end_time - start_time
        lossDir["allParams"] = x
        lossDir["resolution_factor"] = resolution_factor
        
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

        
        minLoss = 1
        for i in range(len(lossDir)):
            for j in range(len(lossDir[i])):
                if lossDir[i][j]["lossTotal"] <= minLoss:
                    minLoss = lossDir[i][j]["lossTotal"]
                    opt = lossDir[i][j]["allParams"]

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
        resultDict["minLoss"] = minLoss
        resultDict["opt_params"] = opt
        resultDict["time_min"] = (end - start) / 60
        
        return tumor, resultDict

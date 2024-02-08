from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver as fwdSolver
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
    def __init__(self, settings, csf, diffusionTensors, edema, enhancing, necrotic):
        self.settings = settings
        self.edema = edema
        self.enhancing = enhancing
        self.necrotic = necrotic
        self.csf = csf
        self.diffusionTensors = diffusionTensors


    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):

        lambdaFlair = self.settings["lossLambdaFlair"]
        lambdaT1c = self.settings["lossLambdaT1"]

        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c	)
        lossFlair = 1 - dice(proposedEdema, self.edema)
        lossT1c = 1 - dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c 

        #catch none values
        if not loss<=1:
            loss = 1

        return loss, {"lossFlair":lossFlair ,"lossT1c": lossT1c,  "lossTotal":loss}

    def forward(self, x, resolution_factor =1.0):
        values = []
        for key in self.fullVariableList:
            if key in self.settings["fixedParameters"]:
                values.append(self.settings[key])
            else:
                values.append(x[self.variableList.index(key)])
        
        parameters = {
            'Dw': values[self.fullVariableList.index("Dw")],         # Diffusion coefficient for white matter
            'rho': values[self.fullVariableList.index("Dw")],        # Proliferation rate
            "diffusionEllipsoidScaling": values[self.fullVariableList.index("diffusionEllipsoidScaling")],
            "diffusionTensorExponent": values[self.fullVariableList.index("diffusionTensorExponent")],
            'NxT1_pct': values[self.fullVariableList.index("NxT1_pct")], 
            'NyT1_pct': values[self.fullVariableList.index("NyT1_pct")],
            'NzT1_pct': values[self.fullVariableList.index("NzT1_pct")],
            'diffusionTensors': self.diffusionTensors,
            'resolution_factor':resolution_factor
        }
        print("run: ", x)
        solver = fwdSolver(parameters)
        tumor = solver.solve()["final_state"]


    def getLoss(self, x, gen):
        #print('Debug get Loss')

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


        values = []
        for key in self.fullVariableList:
            if key in self.settings["fixedParameters"]:
                values.append(self.settings[key])
            else:
                values.append(x[self.variableList.index(key)])
        
        parameters = {
            'Dw': values[self.fullVariableList.index("Dw")],         # Diffusion coefficient for white matter
            'rho': values[self.fullVariableList.index("Dw")],        # Proliferation rate
            "diffusionEllipsoidScaling": values[self.fullVariableList.index("diffusionEllipsoidScaling")],
            "diffusionTensorExponent": values[self.fullVariableList.index("diffusionTensorExponent")],
            'NxT1_pct': values[self.fullVariableList.index("NxT1_pct")], 
            'NyT1_pct': values[self.fullVariableList.index("NyT1_pct")],
            'NzT1_pct': values[self.fullVariableList.index("NzT1_pct")],
            'diffusionTensors': self.diffusionTensors,
            'resolution_factor':resolution_factor
        }
        print("run: ", x)
        #print('Debug start sovler')
        solver = fwdSolver(parameters)

        
        #print('Debug start solve run')
        tumor = solver.solve()["final_state"]

        #print('Debug end solve run')
        
        thresholdT1c = values[self.fullVariableList.index("thresholdT1c")]	
        thresholdFlair = values[self.fullVariableList.index("thresholdFlair")]
        loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
        end_time = time.time()

        lossDir["time"] = end_time - start_time
        lossDir["allParams"] = x
        lossDir["resolution_factor"] = resolution_factor
        
        print("loss: ", loss, "lossDir: ", lossDir, "x: ", x)

        return loss, lossDir

    def run(self):
        start = time.time()

        self.fullVariableList = ["NxT1_pct", "NyT1_pct", "NzT1_pct", "Dw", "rho","diffusionEllipsoidScaling","diffusionTensorExponent","thresholdT1c","thresholdFlair"]

        self.variableList, self.fixedList = [], []
        for key in self.fullVariableList:
            if key in self.settings["fixedParameters"]:
                self.fixedList.append(key)
            else:
                self.variableList.append(key)

        initValues, parameterRanges = [], []
        for key in self.variableList:
            initValues.append(self.settings[key])
            parameterRanges.append(self.settings[key + "_range"])

        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], self.settings["generations"], workers=self.settings["workers"], trace=True, parameterRange=parameterRanges)

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
        
        # forward run with optimal parameters
        values = []
        for key in self.fullVariableList:
            if key in self.settings["fixedParameters"]:
                values.append(self.settings[key])
            else:
                values.append(opt[self.variableList.index(key)])
        
        parameters = {
            'Dw': values[self.fullVariableList.index("Dw")],         # Diffusion coefficient for white matter
            'rho': values[self.fullVariableList.index("Dw")],        # Proliferation rate
            "diffusionEllipsoidScaling": values[self.fullVariableList.index("diffusionEllipsoidScaling")],
            "diffusionTensorExponent": values[self.fullVariableList.index("diffusionTensorExponent")],
            'NxT1_pct': values[self.fullVariableList.index("NxT1_pct")], 
            'NyT1_pct': values[self.fullVariableList.index("NyT1_pct")],
            'NzT1_pct': values[self.fullVariableList.index("NzT1_pct")],
            'diffusionTensors': self.diffusionTensors,
            'resolution_factor':1
        }
        
        solver = fwdSolver(parameters)
        tumor = solver.solve()["final_state"]

        end = time.time()

        resultDict = {}

        resultDict["fixedParameters"] = self.fixedList
        resultDict["variableParameters"] = self.variableList
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

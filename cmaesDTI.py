from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver as fwdSolverDTI
from TumorGrowthToolkit.FK import Solver as fwdSolverFK
import cmaes
import numpy as np
import nibabel as nib
import time
import wandb 
import matplotlib.pyplot as plt
from scipy import ndimage

def dice(a, b):
    boolA, boolB = a > 0, b > 0 
    if np.sum(boolA) + np.sum(boolB) == 0:
        return 0

    return 2 * np.sum( np.logical_and(boolA, boolB)) / (np.sum(boolA) + np.sum(boolB))

class CmaesSolver():
    def __init__(self, settings, diffusionTensors, edema, enhancing, necrotic, gm = None, wm=None, doLog = True):

        self.doLog = doLog

        self.settings = settings
        self.edema = edema
        self.enhancing = enhancing
        self.necrotic = necrotic
        self.diffusionTensors = diffusionTensors

        self.init_scale = 1.0

        if gm is not None and wm is not None:
            self.gm = gm
            self.wm = wm

        self.fullVariableList = ["NxT1_pct", "NyT1_pct", "NzT1_pct", "Dw", "rho","diffusionEllipsoidScaling","diffusionTensorExponent","thresholdT1c","thresholdFlair", "stopping_volume", "stopping_time"]

        self.minLoss = np.inf
    
    def logImges(self, tumor):
        com = ndimage.center_of_mass(tumor)
        try:
            z = int(com[2])
        except:
            z = 0
            print("Error in center of mass, using z=0, tumor  all zeros")
        brainmask = np.sum(np.sum(self.diffusionTensors, axis=-1),axis=-1) 
        tissue = brainmask*0.2 +self.edema *0.3 + self.enhancing * 0.6 + self.necrotic * 0.8
        
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#FBB760", "#F00F0F"]  # RGB values for orange and red
        n_bins = 100  # Number of bins for the color map
        cmap_name = 'orange_red'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        plt.imshow(tissue[:,:,z],alpha=0.5, cmap='gray')
        plt.imshow(tumor[:,:,z], alpha=0.5*(tumor[:,:,z]>0.01), cmap = cmap, vmin=0, vmax=1)	

        #plt.title('Tumor')
        #plt.colorbar()
        if self.doLog:
            wandb.log({"tumor": wandb.Image(plt)})

    def lossfunction(self, tumor, thresholdT1c, thresholdFlair):

        lambdaFlair = self.settings["lossLambdaFlair"]
        lambdaT1c = self.settings["lossLambdaT1"]

        proposedEdema = np.logical_and(tumor > thresholdFlair, tumor < thresholdT1c	)
        diceFlair = dice(proposedEdema, self.edema)
        diceT1c = dice(tumor > thresholdT1c, np.logical_or(self.necrotic, self.enhancing))
        lossFlair = 1 - diceFlair
        lossT1c = 1 - diceT1c
        loss = lambdaFlair * lossFlair + lambdaT1c * lossT1c 

        #catch none values
        if not loss<=1:
            loss = 1

        return loss, {"lossFlair":lossFlair ,"lossT1c": lossT1c,  "lossTotal":loss, "diceFlair":diceFlair, "diceT1c":diceT1c}


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
            'rho': values[self.fullVariableList.index("rho")],        # Proliferation rate
            "diffusionEllipsoidScaling": values[self.fullVariableList.index("diffusionEllipsoidScaling")],
            "diffusionTensorExponent": values[self.fullVariableList.index("diffusionTensorExponent")],
            'NxT1_pct': values[self.fullVariableList.index("NxT1_pct")], 
            'NyT1_pct': values[self.fullVariableList.index("NyT1_pct")],
            'NzT1_pct': values[self.fullVariableList.index("NzT1_pct")],
            'diffusionTensors': self.diffusionTensors,
            'resolution_factor':resolution_factor,
            'stopping_volume': values[self.fullVariableList.index("stopping_volume")],
            'stopping_time': values[self.fullVariableList.index("stopping_time")],
            'init_scale': self.init_scale,
            'verbose': True
        }

        if self.settings["runNormalFKInsteadOfDTI"]:
            parameters["diffusionTensors"] = None
            parameters["difffusionEllipsoidScaling"] = None
            parameters["diffusionTensorExponent"] = None
            parameters["gm"] = self.gm
            parameters["wm"] = self.wm
            fwdSolver = fwdSolverFK
        else:
            fwdSolver = fwdSolverDTI

        #print("run: ", x)
        #print('Debug start sovler')
        solver = fwdSolver(parameters)

        input_parameters = parameters.copy()
        del input_parameters['diffusionTensors']
        if self.settings["runNormalFKInsteadOfDTI"]:
            del input_parameters['gm']
            del input_parameters['wm']
        print("-------------------")
        print("input_parameters: ", input_parameters)
        print("-------------------")
        if self.doLog:
            wandb.log({"input_parameters": input_parameters})
        
        #print('Debug start solve run')
        results = solver.solve()
        thresholdT1c = values[self.fullVariableList.index("thresholdT1c")]	
        thresholdFlair = values[self.fullVariableList.index("thresholdFlair")]
        if results["success"] == True:

            tumor = results["final_state"]

            if self.doLog:
                self.logImges(tumor)


            loss, lossDir = self.lossfunction(tumor, thresholdT1c, thresholdFlair)
        else:
            loss = 1
            lossDir = {"lossFlair":1 ,"lossT1c": 1,  "lossTotal":1, "diceFlair":0, "diceT1c":0}

        end_time = time.time()

        lossDir["time"] = end_time - start_time
        lossDir["allParams"] = x
        lossDir["input_parameters"] = input_parameters
        lossDir["resolution_factor"] = resolution_factor

        if results["success"] == True:
            del results["initial_state"]
            del results["final_state"]
            try:
                del results["time_series"]
            except:
                print("no time series")

        lossDir["results"] = results
        lossDir["thresholdT1c"] = thresholdT1c
        lossDir["thresholdFlair"] = thresholdFlair
  
        print( "lossDir: ", lossDir)
    
        if self.doLog:
            wandb.log(lossDir)


        if loss < self.minLoss:
            self.minLoss = loss
            self.opt = x
            self.opt_lossDir = lossDir

            #TODO self.saveResults()


        return loss, lossDir

    def run(self):
        start = time.time()

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

        if self.doLog:
            wandb.init(project="evolutionary_sampling")
            wandb.config.update(self.settings)

        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], self.settings["generations"], workers=self.settings["workers"], trace=True, parameterRange=parameterRanges, doLog=self.doLog)

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
            'rho': values[self.fullVariableList.index("rho")],        # Proliferation rate
            "diffusionEllipsoidScaling": values[self.fullVariableList.index("diffusionEllipsoidScaling")],
            "diffusionTensorExponent": values[self.fullVariableList.index("diffusionTensorExponent")],
            'NxT1_pct': values[self.fullVariableList.index("NxT1_pct")], 
            'NyT1_pct': values[self.fullVariableList.index("NyT1_pct")],
            'NzT1_pct': values[self.fullVariableList.index("NzT1_pct")],
            'diffusionTensors': self.diffusionTensors,
            'resolution_factor':1,
            'stopping_volume': values[self.fullVariableList.index("stopping_volume")],
            'stopping_time': values[self.fullVariableList.index("stopping_time")],
            'init_scale': self.init_scale
        }
        
        # ugly... but works	
        if self.settings["runNormalFKInsteadOfDTI"]:
            parameters["diffusionTensors"] = None
            parameters["difffusionEllipsoidScaling"] = None
            parameters["diffusionTensorExponent"] = None
            parameters["gm"] = self.gm
            parameters["wm"] = self.wm
            fwdSolver = fwdSolverFK
        else:
            fwdSolver = fwdSolverDTI
        solver = fwdSolver(parameters)
        tumor = solver.solve()["final_state"]
        del solver

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
        resultDict["settings"] = self.settings

        if self.doLog:
            wandb.log(resultDict)

            wandb.finish()
        
        return tumor, resultDict

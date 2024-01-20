
import cmaes
import numpy as np
import nibabel as nib
import time
import nibabel as nib
from TumorGrowthToolkit.FK_2c import Solver
import time

def dice(true_mask, pred_mask):
    intersection = np.sum(true_mask & pred_mask)
    return 2. * intersection / (np.sum(true_mask) + np.sum(pred_mask))

def create_segmentation_map(P, N, th_necro_n, th_enhancing_p, th_edema_p):
    segmentation_map = np.zeros(P.shape, dtype=int)
    edema_mask = (P >= th_edema_p) & (P < th_enhancing_p)
    segmentation_map[edema_mask] = 3
    enhancing_core_mask = P >= th_enhancing_p
    segmentation_map[enhancing_core_mask] = 1
    necrotic_core_mask = N > th_necro_n
    segmentation_map[necrotic_core_mask] = 4
    return segmentation_map

def writeNii(arrays_dict, base_path="", affine=np.eye(4)):
    for key, array in arrays_dict.items():
        # Construct the file path using the key and base path
        if base_path == "":
            path = f"{key}_%dx%dx%dle.nii.gz" % array.shape
        else:
            path = f"{base_path}_{key}.nii.gz"

        # Create and save the NIfTI image
        nibImg = nib.Nifti1Image(array, affine)
        nib.save(nibImg, path)


class CmaesSolver():
    def __init__(self,  settings, wm, gm, edema, enhancing, pet, necrotic):
        self.settings = settings
        self.wm = wm
        self.gm = gm
        self.edema = edema
        self.enhancing = enhancing
        self.pet = pet
        self.necrotic = necrotic


    def loss_function(self, tumor, thresholdT1c, thresholdFlair, thresholdNecro):
        lambdaFlair = 0.150
        lambdaT1c = 0.350
        lambdaPET = 0.250
        lambdaNecro = 0.350

        # Create segmentation map
        segmentation_map = create_segmentation_map(tumor['P'], tumor['N'], thresholdNecro, thresholdT1c, thresholdFlair)

        # Compute dice scores for different regions
        lossEdema = 1 - dice(segmentation_map == 3, self.edema)
        lossEnhancing = 1 - dice(segmentation_map == 1, self.enhancing)
        lossNecrotic = 1 - dice(segmentation_map == 4, self.necrotic)

        # PET correlation
        petInsideTumorRegion = self.pet * np.logical_or(self.edema, self.enhancing)
        lossPet = 1 - np.corrcoef(tumor['P'].flatten(), petInsideTumorRegion.flatten())[0, 1]

        # Weighted loss
        loss = (lambdaFlair * lossEdema +
                lambdaT1c * lossEnhancing +
                lambdaNecro * lossNecrotic +
                lambdaPET * lossPet)

        # Catch non-valid values
        loss = min(loss, 1)

        return loss, {
            "lossEdema": lossEdema,
            "lossEnhancing": lossEnhancing,
            "lossNecrotic": lossNecrotic,
            "lossPet": lossPet,
            "lossTotal": loss
        }


    def forward(self, x, resolution_factor=0.5):
        parameters = {
            'Dw': x[4],
            'rho': x[3],
            'lambda_np': x[5],
            'sigma_np': x[6],
            'D_s': x[7],
            'lambda_s': x[8],
            'gm': self.gm,
            'wm': self.wm,
            'NxT1_pct': x[0],
            'NyT1_pct': x[1],
            'NzT1_pct': x[2],
            'resolution_factor': resolution_factor
        }
        print("run: ", x)
        solver = Solver(parameters)
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
        
        tumor = self.forward(x[:-3],resolution_factor)
        
        thresholdFlair = x[-3]
        thresholdT1c = x[-2]
        thresholdNecro = x[-1]
        loss, lossDir = self.loss_function(tumor, thresholdT1c, thresholdFlair, thresholdNecro)
        end_time = time.time()

        lossDir["time"] = end_time - start_time
        lossDir["allParams"] = x
        lossDir["resolution_factor"] = resolution_factor
        
        print("loss: ", loss, "lossDir: ", lossDir, "x: ", x)

        return loss, lossDir

    def run(self):
        start = time.time()
        
        initValues = (
            self.settings["NxT1_pct0"], self.settings["NyT1_pct0"], self.settings["NzT1_pct0"], 
            self.settings["dw0"], self.settings["rho0"], 
            self.settings["lambda_np0"], self.settings["sigma_np0"], self.settings["D_s0"], 
            self.settings["lambda_s0"], self.settings["thresholdT1c0"], self.settings["thresholdFlair0"], self.settings["thresholdNecro0"]
        )


        trace = cmaes.cmaes(self.getLoss, initValues, self.settings["sigma0"], self.settings["generations"], workers=self.settings["workers"], trace=True, parameterRange=self.settings["parameterRanges"])

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
                if lossDir[i][j]["lossTotal"] < minLoss:
                    minLoss = lossDir[i][j]["lossTotal"]
                    opt = lossDir[i][j]["allParams"]

        tumor = self.forward(opt)
        end = time.time()

        resultDict = {
            "nsamples": nsamples,
            "y0s": y0s,
            "xs0s": xs0s,
            "sigmas": sigmas,
            "Cs": Cs,
            "pss": pss,
            "pcs": pcs,
            "Cmus": Cmus,
            "C1s": C1s,
            "xmeans": xmeans,
            "lossDir": lossDir,
            "minLoss": minLoss,
            "opt_params": opt,
            "time_min": (end - start) / 60
        }
        
        return tumor, resultDict

# Glioma Inverse Fitting Tool

With this tool, you can fit a tumor growth model to patient data. 

<img width="1029" alt="image" src="https://github.com/jonasw247/cmaesForPythonFWD/assets/13008145/af0a2b44-2dd3-451e-911e-963766c63eba">

Convolution Matrix Adaptation Evolution Strategy (CMA-ES) is used to find the optimal data-fitting parameters. The patient data can be fitted to various tumor models e.g., Fisher-Kolmogorov (reaction-diffusion) or Fisher-Kolmogorov with a nutrient field. For forward modeling the TumorGowthToolkit is used (https://github.com/m1balcerak/TumorGrowthToolkit).

# Tutorial 
Run the **tutorialFisherKolmogorov.ipynb** to see how to use the tool.

# Overview

### cmaes.py
is a Python module implementing the plain Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

### cmaesForFWD.py
is a Python module implementing the CMA-ES for a specific forward solver. Defining the loss function and the forward solver is required.

### analyse.py 
is a Python module implementing the analysis of the results of the CMA-ES.

# Description 
![image](https://github.com/jonasw247/glioma-inverse-fitting-tool/assets/13008145/76f9145a-dfaa-4a4a-89e1-e2ab7d4815e7)

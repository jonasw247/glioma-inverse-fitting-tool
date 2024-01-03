# Glioma Inverse Fitting Tool (GIFT)

With this tool you can fit a tumor growth model to patient data. 

<img width="1029" alt="image" src="https://github.com/jonasw247/cmaesForPythonFWD/assets/13008145/c1094a67-890b-4c28-91a1-efdd96926b45">


Convolution Matrix Adaptation Evolution Strategy (CMA-ES) is used to find the optimal parameters fitting the data. The patient Data can be fitted to various tumor models e.g. Fisher-Kolmogorov (reaction diffusion) or Fisher-Kolmogorov with a nutrient field. For forward modelling the TumorGowthToolkit is used (https://github.com/m1balcerak/TumorGrowthToolkit).

# Tutorial 
Run  the tutorialFisherKolmogorov.ipynb to see how to use the tool.

# Overview

cmaes.py is a Python module implementing the plain Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

cmaesForFWD.py is a Python module implementing the CMA-ES for a specific forward solver. Defining the loss function and the forward solver is required.

analyse.py is a Python module implementing the analysis of the results of the CMA-ES.

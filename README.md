# Diffusion-HMC

# Abstract
Diffusion generative models have excelled at diverse image generation and reconstruction tasks across fields. A less explored avenue is their application to discriminative tasks involving regression or classification problems. The cornerstone of modern cosmology is the ability to generate predictions for observed astrophysical fields from theory and constrain physical models from observations using these predictions. This work uses a single diffusion generative model to address these interlinked objectives -- as a surrogate model or emulator for cold dark matter density fields conditional on input cosmological parameters, and as a parameter inference model that solves the inverse problem of constraining the cosmological parameters of an input field. The model is able to emulate fields with summary statistics consistent with those of the simulated target distribution. 

We then leverage the approximate likelihood of the diffusion generative model to derive tight constraints on cosmology by using the Hamiltonian Monte Carlo method to sample the posterior on cosmological parameters for a given test image. Finally, we demonstrate that this parameter inference approach is more robust to the addition of noise than baseline parameter inference networks.

# Directory Structure
```
diffusion-hmc
│   README.md
└───annotated
│   │   main.py # Code to train the model
│   │   evaluate.py # Evaluation / Analysis utilities
│   └───hf_diffusion # Directory with the diffusion model code
│   │   hmc_inference.py # Parameter inference using an HMC
│   │   compute_likelihoods.py # Grid-based parameter inference
│   └───config # Directory with training config files
│   │   classifier_scripts  # Directory with code needed for baseline parameter inference network comparisons
|
└───notebooks
    │   
    │   
```
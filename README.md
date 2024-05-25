# Diffusion-HMC: Parameter Inference with Diffusion Model driven Hamiltonian Monte Carlo
[![arXiv](https://img.shields.io/badge/arXiv-2405.05255%20-purple.svg)](https://arxiv.org/abs/2405.05255)

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
    │   Figure1_LH.ipynb: Figure 1
    │   Figures_HMC_Clean.ipynb: Parameter Inference Plots
    │   Figure_Robustness.ipynb: Figure 7: Robustness to Noise
```

# Citation
If you found this repo or our paper useful / relevant:
```
@article{mudur2024diffusion,
  title={Diffusion-HMC: Parameter Inference with Diffusion Model driven Hamiltonian Monte Carlo},
  author={Mudur, Nayantara and Cuesta-Lazaro, Carolina and Finkbeiner, Douglas P},
  journal={arXiv preprint arXiv:2405.05255},
  year={2024}
}
```

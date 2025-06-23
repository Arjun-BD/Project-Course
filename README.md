# Experimentation on [DPAR: Decoupled Graph Neural Networks with Node-Level Differential Privacy](https://www2024.thewebconf.org/accepted/research-tracks/)

The different branches contain the different experiments conducted:

- `Abhinav` contains experiments using the feature aware random walk and the graph sampling experiments.
- `Arjun` contains experiments using the clustering method.
- `SALSA` and `SALSA_revamp` contain other experiments replacing the Personalised Pagreank Algorithm with Gravity Equation and Heat based paradigms.

  The base code for this repository is taken from the official implementation of DPAR by the paper authors.

  Note that before running the scripts, the `METIS_DLL` environment variable has to be set to the appropriate file (The `metis.dll` file is provided in the `Metis` directory) 

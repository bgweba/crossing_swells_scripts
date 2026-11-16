# Crossing Swells Analysis Scripts

This repository contains the Python scripts used for the analysis of crossing swells in the Agulhas Current region, as described in the manuscript: **Influence of the Agulhas Current on crossing swells along the South African coast**.

The scripts process WW3 (WAVEWATCH III) model output (both with and without surface currents) to generate the spatial maps, spectral decompositions, and percentage tables presented in the manuscript.

## ⚙️ Environment Setup & Dependencies

Because the scripts were run on a server, the environment setup requires both `pip` and `conda`. 

**1. Standard Libraries (via pip):**
Install the standard dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt

# ADME Benchmarks

Scripts for benchmarking models predicting ADME properties from molecular structure.

## Models
Currently benchmarked models:
- Random Forest
- Graph Attention Network (GAT)
- Graph Convulational Nework (GCN)

Model implementations from DeepChem. Models are largely trained using default DeepChem parameters without hyperparameter/model architecture optimisation. This is an important next step.

Molecular featurization is done by ECFP for the RF model, and a graph convolution featurizer for GAT and GCN.

## Datasets
Datasets are sourced from Therapeutics Data Commons (TDC). The following currently used for benchmarking:
- Caco-2 (Cell Effective Permeability), Wang et al. (regression)
- Half Life, Obach et al. (regression)
- BBB (Blood-Brain Barrier), Martins et al. (classification)
- CYP P450 2C19 Inhibition, Veith et al. (classification)

Chosen based on breadth (covering ADME domains) and task type (classification and regression), as well as importance for drug discovery. Pipeline can be arbritarily expanded to cover all datasets, or refined based on customer interests.

## Installation

Most important dependencies are `python 3.10, tdc, pytorch, deepchem`.

Full dependencies can be installed from `environment.yml`:

```conda env create -f environment.yml```


## Usage
See `benchmarking.ipynb` for full usage. 
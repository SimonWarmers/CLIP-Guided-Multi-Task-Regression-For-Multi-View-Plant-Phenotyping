# CLIP-GUIDED MULTI-TASK REGRESSION FOR MULTI-VIEW PLANT PHENOTYPING

## Repository Structure

### Notebooks
The experiments are organized into two approaches:

- **`unimodal/`** – Multi-task Unimodal Baseline experiments for each plant type:
  - `mustard.ipynb`, `radish.ipynb`, `wheat.ipynb`

- **`multimodal/`** – Level-aware Multimodal Fusion experiments for each plant type:
  - `mustard.ipynb`, `radish.ipynb`, `wheat.ipynb`
  - `Level_training/` – Contains the level MLP training notebook and trained model

## Installation

The required dependencies can be found in the `requirements.txt` file.

## Dataset

The real images must be downloaded from the **GROMO25** dataset:

🔗 **https://data.annam.ai/gromo25/**

The `data/` folder contains the refined CSV files for each plant type.

## Results

The file `results.csv` contains the performance metrics from which the tables in the paper are derived. These results are computed by running the notebooks.

## Supplementary Material

- **`custom_lime.py`** – Custom LIME implementation used for model interpretability (supplementary material)
- The custom LIME is used to create plots at the end of the notebooks.

## Citation
If you use this code in your research, please cite: 
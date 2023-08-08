# EHIGN_SBVS

## Dataset
The LIP-PCBA dataset is publicly available at the following locations:

- **Original LIT-PCBA [1]:** [Link](https://drugdesign.unistra.fr/LIT-PCBA/)
- **Docked Data (with 3D structures for small compounds) [2]:** [Link](https://zenodo.org/record/4291725#.X7-JHTOUMyg)

Preprocessed data (molecular graphs) can be downloaded from:

- [Link 1](https://zenodo.org/record/8208800)
- [Link 2](https://zenodo.org/record/8219837)

## Requirements
The following Python packages are required:  
dgl==0.9.0  
networkx==2.5  
numpy==1.19.2  
pandas==1.1.5  
pymol==0.1.0  
rdkit==2022.3.5  
scikit_learn==1.1.2  
scipy==1.5.2  
torch==1.10.2  
tqdm==4.63.0  
openbabel==3.3.1 (conda install -c conda-forge openbabel)

## Structure and Descriptions

### Directories
- **`./config`:** Parameters used in EHIGN.
- **`./log`:** Logger.
- **`./model`:** Contains several trained models for reproducing results.

### Files
- **`CIGConv.py`, `NIGConv.py`, `EHIGN.py`:** Implementations of CIGConv, NIGConv, and EHIGN.
- **`HGC.py`:** Heterogeneous graph neural network implementation (modified from dgl source code).
- **`preprocess_complex.py`:** Prepare input complexes.
- **`graph_constructor.py`:** Convert protein-ligand complexes into heterogeneous graphs.
- **`train.py`:** Train the EHIGN model.
- **`test.py`:** Use models in ./model directory for prediction.

## Step-by-step Running

### Organize the Data
Organize the data as follows:  

-docking_poses  
&ensp;&ensp;-ALDH1_4x4l  
&ensp;&ensp;&ensp;&ensp; -train  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; -ALDH1_4x4l_decoys_22407376-EHIGN.dgl  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; ...  
&ensp;&ensp;&ensp;&ensp; -val  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; ...  
&ensp;&ensp;-FEN1_5fv7  
&ensp;&ensp;&ensp;&ensp; ...  
&ensp;&ensp;-GBA_2v3e  
&ensp;&ensp;&ensp;&ensp; ...  
...  

You can download the processed data from https://zenodo.org/record/8208800 and https://zenodo.org/record/8219837

### 1. Reproduce the reported results
The ./model directory contains seven trained models that can be used to reproduce the reported results.

### 2. Model training
python train.py --data_root your_own_data_path/docking_poses

### 3. Model testing
python test.py --data_root your_own_data_path/docking_poses  
By default, this will use the seven trained models in the ./model directory to predict.

### 4. Process raw data
python preprocess_complex.py --data_root your_own_data_path/docking_poses  
python graph_constructor.py --data_root your_own_data_path/docking_poses  

## Reference
[1] Tran-Nguyen V K, Jacquemard C, Rognan D. LIT-PCBA: an unbiased data set for machine learning and virtual screening[J]. Journal of chemical information and modeling, 2020, 60(9): 4263-4273.  
[2] Shen C, Weng G, Zhang X, et al. Accuracy or novelty: what can we gain from target-specific machine-learning-based scoring functions in virtual screening?[J]. Briefings in Bioinformatics, 2021, 22(5): bbaa410.



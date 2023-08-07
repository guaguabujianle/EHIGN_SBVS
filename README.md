# EHIGN_SBVS

## Dataset
The LIP-PCBA dataset is publicly available and can be accessed here:
- Original LIT-PCBA [1]: https://drugdesign.unistra.fr/LIT-PCBA/
- Docked data (with 3D structures for small compounds) [2]: https://zenodo.org/record/4291725#.X7-JHTOUMyg  

You can download the preprocessed data (molecular graphs) from:  
https://zenodo.org/record/8208800  
https://zenodo.org/record/8219837  

## Requirements  
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

## Descriptions of folders and files
+ **./config**: Parameters used in EHIGN, you should change the "data_root" in TrainConfig.json to your own one.
+ **./log**: Logger.
+ **./model**: This dir contains several trained models that can be used to reproduce the reported results.
+ **CIGConv.py**: The implementation of CIGConv.
+ **NIGConv.py**: The implementation of NIGConv.
+ **EHIGN.py**: The implementation of EHIGN.
+ **HGC.py**: The implementation of the heterogeneous graph neural network, where most of the contents are copied from the source code of dgl, but we have made some modifications so that it can process edge features.
+ **preprocess_complex.py**: Prepare input complexes. First, you should download the data from https://zenodo.org/record/4291725#.X7-JHTOUMyg. Then you should replace the data_root with your own path. Finally, run "python preprocess_complex.py"
+ **graph_constructor.py**: Convert protein-ligand complexes into heterogeneous graphs. You should replace the data_root with your own path.
+ **train.py**: Train EHIGN model. Change config/TrainConfig.json before running it.
+ **test.py**, By default, this will use the seven trained models in ./model dir to predict.

## Step-by-step running: will be available soon.  
### 1. Reproduce the reported results
The ./model directory contains seven trained models that can be used to reproduce the reported results.

### 2. Model training

### 3. Model testing

### 4. Process raw data

### 5. Test the trained model in other external test sets

## Reference
[1] Tran-Nguyen V K, Jacquemard C, Rognan D. LIT-PCBA: an unbiased data set for machine learning and virtual screening[J]. Journal of chemical information and modeling, 2020, 60(9): 4263-4273.  
[2] Shen C, Weng G, Zhang X, et al. Accuracy or novelty: what can we gain from target-specific machine-learning-based scoring functions in virtual screening?[J]. Briefings in Bioinformatics, 2021, 22(5): bbaa410.



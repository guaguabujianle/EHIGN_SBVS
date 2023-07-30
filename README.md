# EHIGN_SBVS

## Dataset
The LIP-PCBA dataset is publicly available and can be accessed here:
- Original LIT-PCBA [1]: https://drugdesign.unistra.fr/LIT-PCBA/
- 3D structure available one [2]: https://zenodo.org/record/4291725#.X7-JHTOUMyg  

You can download the preprocessed data from: 

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
+ **./model**: This dir contains serveral trained model that can be use to reproduce the reported results.
+ **CIGConv.py**: The implementation of CIGConv.
+ **NIGConv.py**: The implementation of NIGConv.
+ **EHIGN.py**: The implementation of EHIGN.
+ **HGC.py**: The implementation of the heterogeneous graph neural network, where most of the contents are copied from the source code of dgl, but we have made some modifications so that it can process edge features.
+ **preprocess_complex.py**: Prepare input complexes. First, you should download the data from https://zenodo.org/record/4291725#.X7-JHTOUMyg. Then you should replace the data_root with your own path. Finally, run "python preprocess_complex.py"
+ **graph_constructor.py**: Convert protein-ligand complexes into heterogeneous graphs. You should replace the data_root with your own path.
+ **train.py**: Train EHIGN model. Change config/TrainConfig.json before running it.
+ **test.py**, By default, this will use the seven trained models in ./model dir to predict.

## Step-by-step running:  

### 1. Model training
Firstly, download the preprocessed datasets from https://drive.google.com/file/d/1oGUP4z7htNXyxTqx95HNSDLsaoxa3fX7/view?usp=share_link, and put them into this folder and organize them as './data/train', './data/valid', './data/test2013/', './data/test2016/', and  './data/test2019/'.  
Secondly, run train.py using `python train.py`.  

### 2. Model testing
Run test.py using `python test.py`.    
You may need to modify some file paths in the source code before running it.

### 3. Process raw data
We provide a demo to explain how to process the raw data. This demo use ./data/toy_examples.csv and ./data/toy_set/ as examples.  
Firstly, run preprocess_complex.py using `python preprocess_complex.py`.    
Secondly, run graph_constructor.py using `python graph_constructor.py`.  
Thirdly, run train.py using `python train_example.py`.    

### 4. Test the trained model in other external test sets
Firstly, please organize the data as a structure similar to './data/toy_set' folder.  
-data  
&ensp;&ensp;-external_test  
&ensp; &ensp;&ensp;&ensp; -pdb_id  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_ligand.mol2  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_protein.pdb  
Secondly, run preprocess_complex.py using `python preprocess_complex.py`.  
Thirdly, run graph_constructor.py using `python graph_constructor.py`.  
Fourth, run test.py using `python test.py`.  
You may need to modify some file paths in the source code before running it.  

### 5. Cold start settings
The datasets for the cold start settings can be found in the './cold_start_data' folder. These datasets are created from the original training set, taking into account structural differences. If you have already processed the original training set and placed it in the './data/train' folder, you can directly execute the 'train_random.py', 'train_scaffold.py', and 'train_sequence.py' scripts.

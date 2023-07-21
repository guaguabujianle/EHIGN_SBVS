# %%
import os
import shutil
from rdkit import Chem
from tqdm import tqdm
import pickle
import pymol
from rdkit import RDLogger
from utils import create_dir
RDLogger.DisableLog('rdApp.*')

# %%
def generate_data(data_root, distance=5):
    # use FEN1_5fv7 as an example
    complex_ids = ['FEN1_5fv7']
    # complex_ids = ['FEN1_5fv7', 'ALDH1_4x4l', 'GBA_2v3e', 'KAT2A_5h84', 'MAPK1_2ojg', 'PKM2_3gr4', 'VDR_3a2j']

    pbar = tqdm(total=len(complex_ids))
    for cid in complex_ids:
        complex_dir = os.path.join(data_root, cid)
        template_dir = os.path.join(complex_dir, f'{cid}_prot')
        prot_path = os.path.join(template_dir, f'{cid}_p.pdb')
        lig_native_path = os.path.join(template_dir, f'{cid}_l.mol2')

        pymol.cmd.load(prot_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid}_l around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

        for data_type in ['T', 'V']:
            if data_type == 'T':
                data_dir = os.path.join(complex_dir, 'train')
            else:
                data_dir = os.path.join(complex_dir, 'val')

            create_dir([data_dir])

            ligand_actives_path = os.path.join(complex_dir, f"{cid}_SP_active_{data_type}.sdf")
            ligand_decoys_path = os.path.join(complex_dir, f"{cid}_SP_inactive_{data_type}.sdf")
            pokcet_path = os.path.join(complex_dir, f'Pocket_{distance}A.pdb')
            
            pocket = Chem.MolFromPDBFile(pokcet_path, removeHs=True)

            actives_compounds = Chem.SDMolSupplier(ligand_actives_path, removeHs=True)
            for ligand in actives_compounds:
                try:
                    ligand_id = ligand.GetProp('_Name')
                except:
                    continue
                save_path = os.path.join(data_dir, f'{cid}_actives_{ligand_id}.dat')
                complex = (pocket, ligand)

                with open(save_path, 'wb') as f:
                    pickle.dump(complex, f)    

            decoys_compounds = Chem.SDMolSupplier(ligand_decoys_path, removeHs=True)
            for ligand in decoys_compounds:
                try:
                    ligand_id = ligand.GetProp('_Name')
                except:
                    continue
                save_path = os.path.join(data_dir, f'{cid}_decoys_{ligand_id}.dat')
                complex = (pocket, ligand)

                with open(save_path, 'wb') as f:
                    pickle.dump(complex, f)    

        pbar.update(1)



if __name__ == '__main__':
    data_root = '/data2/yzd/docking/SBVS/LIT-PCBA/docking_poses'
    generate_data(data_root)


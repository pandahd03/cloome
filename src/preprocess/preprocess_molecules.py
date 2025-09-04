import sys
sys.path.insert(1, r"/system/user/studentwork/seibezed/cloome")

import os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from src.clip import helpers



def morgan_from_smiles(smiles, radius=3, nbits=1024, chiral=True):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nbits, useChirality=chiral)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

def max_morgan_count_based_from_smiles(smiles, nbits=8192, chiral=True):
    mol = Chem.MolFromSmiles(smiles)

    fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nbits, useChirality=chiral)
    arr_bit = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_bit,arr_bit)

    fp_count = AllChem.GetHashedMorganFingerprint(mol, radius=3, nBits=nbits, useChirality=chiral)
    arr_count = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_count,arr_count)

    arr = np.maximum(arr_bit, arr_count)
    return arr


if __name__ == '__main__':

    indir = "<path-to-your-folder>"
    index = "<path-to-your-folder>/cellpainting-index.csv"
    index = os.path.join(indir, index)

    index_1 = r"/publicwork/sanchez_copied/data/cellpainting-test-phenotype-imgpermol-scaffold.csv"

    index_2 = r"/publicwork/sanchez_copied/data/cellpainting-split-test-imgpermol-scaffold.csv"

    outdir = "/publicwork/sanchez/data/"
    outfile_hdf = "morgan_chiral_fps.hdf5"
    outfile_hdf = os.path.join(outdir, outfile_hdf)

    outdir = r"/system/user/studentwork/seibezed/cloome/src/data"
    outfile_hdf = "morgan_chiral_fps_1024_test_zs_molecules_scaffold.hdf5"
    outfile_hdf = os.path.join(outdir, outfile_hdf)

    n_cpus = 60

    csv_1 = pd.read_csv(index_1)
    csv_2 = pd.read_csv(index_2)
    csv = pd.concat([csv_1, csv_2], ignore_index=True)
    print(csv.head)

    csv["ID"] = csv.apply(lambda row: "-".join([str(row["PLATE_ID"]), str(row["WELL_POSITION"]),  str(row["SITE"])]), axis=1)

    ids = csv["ID"]
    smiles = csv["SMILES"]
    
    fps = helpers.parallelize(morgan_from_smiles, smiles, n_cpus)
    columns = [str(i) for i in range(fps[0].shape[0])]

    print("test:"+str(len(fps)))
    print(np.array(fps))

    df = pd.DataFrame(np.array(fps), index=ids, columns=columns)

    df.to_hdf(outfile_hdf, key="df", mode="w")


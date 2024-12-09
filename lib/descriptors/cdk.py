## Adapted from "JN_for_Model_Creation_Optimized_ECFP4.ipynb"

from enum import IntEnum
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from py4j.java_gateway import JavaGateway, Py4JError
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles

from .transformer import DescriptorTransformer

_cdk_jar = Path(__file__).parent.resolve().joinpath("./cdk-2.7.1.jar")


class ECFPID(IntEnum):
    ECFP0 = 1
    ECFP2 = 2
    ECFP4 = 3
    ECFP6 = 4
    FCFP0 = 5
    FCFP2 = 6
    FCFP4 = 7
    FCFP6 = 8


class ECFPTransformer(DescriptorTransformer):
    def __init__(self, integer_id=ECFPID.ECFP4, n_bits=1024):
        self.integer_id = integer_id
        self.n_bits = n_bits

    def fit(self, X=None, y=None):
        gateway: Any = JavaGateway.launch_gateway(
            classpath=str(_cdk_jar),
            die_on_exit=True,
        )

        cdk = gateway.jvm.org.openscience.cdk

        self.circularFingerprinter_ = cdk.fingerprint.CircularFingerprinter(
            int(self.integer_id)
        )
        self.smilesParser_ = cdk.smiles.SmilesParser(
            cdk.DefaultChemObjectBuilder.getInstance()
        )
        self.init_time = time()

        return self

    def transform_one(self, X):
        try:
            mol = self.mol_to_cdk_mol(X)
            ecfp4 = np.array(
                (self.circularFingerprinter_.getBitFingerprint(mol).getSetbits()),
                dtype=int,
            )
            return self.get_bit_vector(ecfp4)
        except Py4JError as e:
            if self.init_time + 10 >= time():
                raise Exception("Restarted too fast") from e

            self.fit()
            return self.transform_one(X)

    def get_feature_names_out(self, input_features=None):
        return [f"x{i}" for i in range(self.n_bits)]

    def mol_to_cdk_mol(self, mol: Mol):
        smiles = MolToSmiles(mol)
        return self.smilesParser_.parseSmiles(smiles)

    @staticmethod
    def get_bit_vector(indexes):
        idx = np.zeros(1024, dtype=np.float_)
        idx[indexes] = 1
        return idx

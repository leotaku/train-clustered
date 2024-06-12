## Adapted from "JN_for_Model_Creation_Optimized_ECFP4.ipynb"

from enum import IntEnum
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from py4j.java_gateway import JavaGateway, Py4JError
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles

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


class ECFPCalc:
    def __init__(self, integer_id=ECFPID.ECFP4):
        self.integer_id = integer_id
        self.init_time = time()

        self.gateway: Any = JavaGateway.launch_gateway(
            classpath=str(_cdk_jar),
            die_on_exit=True,
        )

        cdk = self.gateway.jvm.org.openscience.cdk
        builder = cdk.DefaultChemObjectBuilder.getInstance()
        self.circularFingerprints = cdk.fingerprint.CircularFingerprinter(
            int(integer_id)
        )
        self.smiles_parser = cdk.smiles.SmilesParser(builder)

    def __getstate__(self):
        return self.integer_id

    def __setstate__(self, integer_id):
        self.__init__(integer_id)

    @staticmethod
    def get_bit_vector(indexes):
        idx = np.zeros(1024, dtype=int)  # Just pre-defined as in Naga's code
        idx[indexes] = 1
        return idx

    def mol_to_cdk_mol(self, mol: Mol):
        smiles = MolToSmiles(mol)
        return self.smiles_parser.parseSmiles(smiles)

    def __call__(self, mol: Mol):
        try:
            cdk_mol = self.mol_to_cdk_mol(mol)
            ecfp4 = np.array(
                (self.circularFingerprints.getBitFingerprint(cdk_mol).getSetbits()),
                dtype=int,
            )
            return self.get_bit_vector(ecfp4)
        except Py4JError as e:
            if self.init_time + 10 >= time():
                raise Exception("Restarted too fast") from e

            self.__init__(self.integer_id)
            return self.__call__(mol)

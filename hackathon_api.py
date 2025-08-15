# hackathon_api.py
from dataclasses import dataclass

@dataclass
class Protein:
    """
    Represents a protein sequence for Boltz prediction.
    """
    id: str
    sequence: str
    msa_path: str                      # A3M path (always provided for hackathon)

@dataclass
class SmallMolecule:
    """
    Represents a small molecule/ligand for Boltz prediction.
    """
    id: str
    smiles: str                        # SMILES string for the ligand

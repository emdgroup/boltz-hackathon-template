
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from dataclasses_json import dataclass_json


# Enum for task_type
class TaskType(str, Enum):
    """Enum for valid hackathon task types."""

    PROTEIN_COMPLEX = "protein_complex"
    PROTEIN_LIGAND = "protein_ligand"


@dataclass_json
@dataclass
class Protein:
    """Represents a protein sequence for Boltz prediction."""

    id: str
    sequence: str
    msa: Optional[str] = None  # A3M path (always provided for hackathon)


@dataclass_json
@dataclass
class SmallMolecule:
    """Represents a small molecule/ligand for Boltz prediction."""

    id: str
    smiles: Optional[str] = None  # SMILES string for the ligand


@dataclass_json
@dataclass
class Datapoint:
    """Represents a single hackathon datapoint for Boltz prediction."""

    datapoint_id: str
    task_type: TaskType
    proteins: list[Protein]
    ligands: Optional[list[SmallMolecule]] = None  # We will only have a SINGLE ligand for the allosteric/orthosteric binding challenge
    ground_truth: Optional[dict[str, Any]] = None


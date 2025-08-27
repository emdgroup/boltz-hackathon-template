
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
    msa: str  # A3M path (always provided for hackathon)
    modifications: Optional[Any] = field(default=None)


@dataclass_json
@dataclass
class SmallMolecule:
    """Represents a small molecule/ligand for Boltz prediction."""

    id: str
    smiles: str  # SMILES string for the ligand


@dataclass_json
@dataclass
class Datapoint:
    """Represents a single hackathon datapoint for Boltz prediction."""

    datapoint_id: str
    task_type: TaskType
    proteins: list[Protein]
    ligands: Optional[list[SmallMolecule]] = None
    ground_truth: Optional[dict[str, Any]] = None
    # Optionally, add other fields as needed (e.g., metadata)


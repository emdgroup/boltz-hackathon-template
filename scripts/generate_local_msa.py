from __future__ import annotations

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Constants
CHAIN_INFO_LENGTH = 2
DEFAULT_CHAIN_NAME = "101"


@dataclass
class LocalColabFoldConfig:
    """Configuration for ColabFold search."""

    colabsearch: str
    query_fpath: str
    db_dir: str
    results_dir: str
    mmseqs_path: Optional[str] = None
    db1: str = "uniref30_2302_db"
    db2: Optional[str] = None
    db3: Optional[str] = "colabfold_envdb_202108_db"
    use_env: int = 1
    filter: int = 1
    db_load_mode: int = 0


class A3MProcessor:
    """Processor for A3M file format."""

    def __init__(self, a3m_file: str, out_dir: str) -> None:
        self.out_dir = out_dir
        self.a3m_file = Path(a3m_file)
        self.a3m_content = self._read_a3m_file()
        self.chain_info = self._parse_header()

    def _read_a3m_file(self) -> str:
        """Read A3M file content."""
        return self.a3m_file.read_text()

    def _parse_header(self) -> tuple[list[str], dict[str, tuple[int, int]]]:
        """Parse A3M header to get chain information."""
        first_line = self.a3m_content.split("\n")[0]
        if first_line[0] == "#":
            lengths, oligomeric_state = first_line.split("\t")

            chain_lengths = [int(x) for x in lengths[1:].split(",")]
            chain_names = [
                f"10{x + 1}" for x in range(len(oligomeric_state.split(",")))
            ]

            # Calculate sequence ranges for each chain
            seq_ranges = {}
            for i, name in enumerate(chain_names):
                start = sum(chain_lengths[:i])
                end = sum(chain_lengths[: i + 1])
                seq_ranges[name] = (start, end)
        else:
            chain_names = [DEFAULT_CHAIN_NAME]
            seq_ranges = {DEFAULT_CHAIN_NAME: (0, len(self.a3m_content.split("\n")[1]))}

        return chain_names, seq_ranges

    def _extract_sequence(self, line: str, range_tuple: tuple[int, int]) -> str:
        """Extract sequence for specific range."""
        seq = []
        no_insert_count = 0
        start, end = range_tuple

        for char in line:
            if char.isupper() or char == "-":
                no_insert_count += 1
            # we keep insertions
            if start < no_insert_count <= end:
                seq.append(char)
            elif no_insert_count > end:
                break

        return "".join(seq)

    def _process_sequence_lines(  # noqa: C901
        self,
        lines: list[str],
        seq_ranges: dict[str, tuple[int, int]],
        chain_names: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Process sequence lines to separate pairing and non-pairing sequences."""
        pairing_a3ms = {name: [] for name in chain_names}
        nonpairing_a3ms = {name: [] for name in chain_names}

        current_query = None
        for line in lines:
            if line.startswith("#"):
                continue

            if line.startswith(">"):
                name = line[1:]
                if name in chain_names:
                    current_query = chain_names[chain_names.index(name)]
                elif name == "\t".join(chain_names):
                    current_query = None

                # Add header line to appropriate dictionary
                if current_query:
                    nonpairing_a3ms[current_query].append(line)
                else:
                    for chain_name in chain_names:
                        pairing_a3ms[chain_name].append(line)
                continue

            # Process sequence line
            if not line:
                continue

            if current_query:
                seq = self._extract_sequence(line, seq_ranges[current_query])
                nonpairing_a3ms[current_query].append(seq)
            else:
                for chain_name in chain_names:
                    seq = self._extract_sequence(line, seq_ranges[chain_name])
                    pairing_a3ms[chain_name].append(seq)

        return nonpairing_a3ms, pairing_a3ms
    
    def _get_query_sequences(self, 
                             chain_names: list[str],
                             pairing_a3ms: dict[str, list[str]],
                             nonpairing_a3ms: dict[str, list[str]]) -> dict[str, str]:

        query_sequences = {}
        for chain_name in chain_names:
            # Try to get query from pairing first, then non-pairing
            pairing_lines = pairing_a3ms.get(chain_name, [])
            nonpairing_lines = nonpairing_a3ms.get(chain_name, [])

            if len(pairing_lines) > 1:
                query_sequences[chain_name] = pairing_lines[1]
            elif len(nonpairing_lines) > 1:
                query_sequences[chain_name] = nonpairing_lines[1]
            else:
                query_sequences[chain_name] = ""

        return query_sequences

    def split_sequences(self) -> None:
        """Split A3M file into pairing and non-pairing sequences."""
        out_dir = Path(self.out_dir)
        chain_names, seq_ranges = self.chain_info

        nonpairing_a3ms, pairing_a3ms = self._process_sequence_lines(
            self.a3m_content.split("\n"), seq_ranges, chain_names
        )

        # Extract query sequences for each chain
        query_sequences = self._get_query_sequences(chain_names, pairing_a3ms, nonpairing_a3ms)

        self._write_output_files(out_dir, nonpairing_a3ms, pairing_a3ms, query_sequences)

    def _write_msa_to_csv(
        self,
        csv_file_name: Path,
        query_sequence: str,
        pairing_sequences: list[str],
        nonpairing_sequences: list[str],
    ) -> None:
        """
        Write MSA sequences to a CSV file with query sequence always first.
        
        Args:
            csv_file_name: Path to the output CSV file
            query_sequence: The query sequence (always written with key=0 as first row)
            pairing_sequences: List of pairing MSA sequences (written with keys starting from 1)
            nonpairing_sequences: List of non-pairing MSA sequences (written with key=-1)
        """
        with csv_file_name.open(mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["key", "sequence"])  # Write header
            
            # ALWAYS write query sequence first with key=0
            writer.writerow([0, query_sequence])
            
            # Write pairing sequences with positive keys starting from 1
            for i, seq in enumerate(pairing_sequences, start=1):
                if seq and not seq.startswith(">"):
                    writer.writerow([i, seq])
            
            # Write non-pairing sequences with key=-1
            for seq in nonpairing_sequences:
                if seq and not seq.startswith(">"):
                    writer.writerow([-1, seq])

    def _write_output_files(
        self,
        out_dir: Path,
        nonpairing_a3ms: dict[str, list[str]],
        pairing_a3ms: dict[str, list[str]],
        query_sequences: dict[str, str],
    ) -> None:
        """
        Write split sequences to output files.
        
        This method combines pairing and non-pairing MSAs into a single CSV file per chain,
        ensuring the query sequence is always written first with key=0.
        
        Args:
            out_dir: Output directory for CSV files
            nonpairing_a3ms: Dictionary of non-pairing MSA sequences by chain
            pairing_a3ms: Dictionary of pairing MSA sequences by chain
            query_sequences: Dictionary of query sequences by chain
        """
        out_dir.mkdir(exist_ok=True)

        # Get all unique chain names from both dictionaries
        all_chain_names = list(pairing_a3ms.keys())
        
        # Process each chain and write combined MSA to CSV
        for i, chain_name in enumerate(all_chain_names):
            csv_file_name = out_dir / f"msa_{i}.csv"
            
            # Get sequences from both sources
            pairing_lines = pairing_a3ms.get(chain_name, [])
            nonpairing_lines = nonpairing_a3ms.get(chain_name, [])
            
            # Get the query sequence for this chain
            query_seq = query_sequences.get(chain_name, "")
            
            # Validate that we have a query sequence
            if not query_seq:
                print(f"Warning: No query sequence found for chain {chain_name}")
                continue
            
            # Extract sequences, skipping header at index 0
            # Skip index 1 if it's identical to the query sequence
            pairing_sequences = []
            nonpairing_sequences = []
            
            # Process pairing sequences
            for idx, line in enumerate(pairing_lines):
                if idx == 0:  # Skip header
                    continue
                if idx == 1 and line == query_seq:  # Skip if identical to query
                    continue
                if line and not line.startswith(">"):
                    pairing_sequences.append(line)
            
            # Process non-pairing sequences
            for idx, line in enumerate(nonpairing_lines):
                if idx == 0:  # Skip header
                    continue
                if idx == 1 and line == query_seq:  # Skip if identical to query
                    continue
                if line and not line.startswith(">"):
                    nonpairing_sequences.append(line)
            
            # Write combined MSA to CSV with query sequence always first
            self._write_msa_to_csv(
                csv_file_name,
                query_seq,
                pairing_sequences,
                nonpairing_sequences,
            )


def run_colabfold_search(config: LocalColabFoldConfig) -> str:
    """Run ColabFold search with given configuration."""
    cmd = [config.colabsearch, config.query_fpath, config.db_dir, config.results_dir]

    # Add optional parameters
    if config.db1:
        cmd.extend(["--db1", config.db1])
    if config.db2:
        cmd.extend(["--db2", config.db2])
    if config.db3:
        cmd.extend(["--db3", config.db3])
    if config.mmseqs_path:
        cmd.extend(["--mmseqs", config.mmseqs_path])
    else:
        cmd.extend(["--mmseqs", "mmseqs"])
    if config.use_env:
        cmd.extend(["--use-env", str(config.use_env)])
    if config.filter:
        cmd.extend(["--filter", str(config.filter)])
    if config.db_load_mode:
        cmd.extend(["--db-load-mode", str(config.db_load_mode)])

    # Use subprocess instead of os.system for security
    subprocess.run(cmd, check=True)  # noqa: S603

    # Return the first .a3m file found in results directory
    result_files = list(Path(config.results_dir).glob("*.a3m"))
    if not result_files:
        error_msg = f"No .a3m files found in {config.results_dir}"
        raise FileNotFoundError(error_msg)
    return str(result_files[0])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ColabFold search and A3M processing tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("query_fpath", help="Path to the query FASTA file")
    parser.add_argument("db_dir", help="Directory containing the databases")
    parser.add_argument("results_dir", help="Directory for storing results")

    # Optional arguments
    parser.add_argument(
        "--colabsearch", help="Path to colabfold_search", default="colabfold_search"
    )
    parser.add_argument(
        "--mmseqs_path", help="Path to MMseqs2 binary", default="mmseqs"
    )
    parser.add_argument("--db1", help="First database name", default="uniref30_2302_db")
    parser.add_argument("--db2", help="Templates database")
    parser.add_argument(
        "--db3", help="Environmental database (default: colabfold_envdb_202108_db)"
    )
    parser.add_argument(
        "--use_env", help="Use environment settings", type=int, default=1
    )
    parser.add_argument("--filter", help="Apply filtering", type=int, default=1)
    parser.add_argument(
        "--db_load_mode", help="Database load mode", type=int, default=0
    )
    parser.add_argument(
        "--output_split", help="Directory for split A3M files", default=None
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Create configuration from arguments
    config = LocalColabFoldConfig(
        colabsearch=args.colabsearch,
        query_fpath=args.query_fpath,
        db_dir=args.db_dir,
        results_dir=args.results_dir,
        mmseqs_path=args.mmseqs_path,
        db1=args.db1,
        db2=args.db2,
        db3=args.db3,
        use_env=args.use_env,
        filter=args.filter,
        db_load_mode=args.db_load_mode,
    )

    # Run search
    results_a3m = run_colabfold_search(config)

    processor = A3MProcessor(results_a3m, args.results_dir)
    if len(processor.chain_info) == CHAIN_INFO_LENGTH:
        processor.split_sequences()



if __name__ == "__main__":
    args = parse_args()
    main(args)

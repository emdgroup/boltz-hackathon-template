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

    def split_sequences(self) -> None:
        """Split A3M file into pairing and non-pairing sequences."""
        out_dir = Path(self.out_dir)
        chain_names, seq_ranges = self.chain_info

        nonpairing_a3ms, pairing_a3ms = self._process_sequence_lines(
            self.a3m_content.split("\n"), seq_ranges, chain_names
        )

        self._write_output_files(out_dir, nonpairing_a3ms, pairing_a3ms)

    def _write_csv_file(
        self, csv_file_name: Path, lines: list[str], start_key: int
    ) -> None:
        """Write sequences to a CSV file."""
        with csv_file_name.open(mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["key", "sequence"])  # Write header

            if lines:
                query_seq = lines[1]
                writer.writerow([0, query_seq])  # Write the query sequence

                # Process remaining sequences
                j_id = 0
                for line in lines[2:]:
                    if line.startswith(">"):
                        pass
                    elif line:
                        writer.writerow(
                            [j_id + start_key, line]
                        )  # Write sequence with incremented key
                        j_id += 1

    def _append_to_csv_file(self, csv_file_name: Path, lines: list[str]) -> None:
        """Append sequences to an existing CSV file."""
        with csv_file_name.open(mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if lines:
                # Process remaining sequences
                for line in lines[2:]:
                    if line.startswith(">"):
                        pass
                    elif line:
                        writer.writerow([-1, line])  # Write sequence with key -1

    def _write_output_files(
        self,
        out_dir: Path,
        nonpairing_a3ms: dict[str, list[str]],
        pairing_a3ms: dict[str, list[str]],
    ) -> None:
        """Write split sequences to output files."""
        out_dir.mkdir(exist_ok=True)

        # Process pairing sequences and write to CSV
        for i, (_, lines) in enumerate(pairing_a3ms.items()):
            csv_file_name = out_dir / f"msa_{i}.csv"
            self._write_csv_file(csv_file_name, lines, 1)

        # Process non-pairing sequences and append to CSV
        for i, (_, lines) in enumerate(nonpairing_a3ms.items()):
            csv_file_name = out_dir / f"msa_{i}.csv"
            self._append_to_csv_file(csv_file_name, lines)


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

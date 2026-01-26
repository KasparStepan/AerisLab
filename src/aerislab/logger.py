"""
CSV logging for simulation state with performance optimization.

Buffers data in memory and writes in batches to minimize I/O overhead.
Implements context manager protocol for safe resource handling.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO

from aerislab.dynamics.body import RigidBody6DOF


class CSVLogger:
    """
    High-performance CSV logger for simulation data.

    Features:
    - Buffered writing (reduces syscalls)
    - Context manager support (safe file handling)
    - Automatic header generation
    - Configurable fields

    Parameters
    ----------
    filepath : str | Path
        Output CSV file path
    buffer_size : int
        Number of rows to buffer before writing. Higher = fewer writes but more memory.
        Recommended: 100-1000 for most simulations.
    fields : List[str] | None
        State fields to log per body. Default: ["p", "q", "v", "w", "f", "tau"]
        Options: "p" (position), "q" (quaternion), "v" (velocity),
                 "w" (angular velocity), "f" (force), "tau" (torque)

    Attributes
    ----------
    filepath : Path
        Path to output CSV file
    buffer_size : int
        Number of rows buffered before flush
    fields : List[str]
        State fields being logged

    Notes
    -----
    **Usage Patterns:**

    1. Context manager (recommended):
    >>> with CSVLogger("output.csv") as logger:
    ...     for step in range(num_steps):
    ...         world.step(...)
    ...         logger.log(world)

    2. Manual management:
    >>> logger = CSVLogger("output.csv")
    >>> for step in range(num_steps):
    ...     logger.log(world)
    >>> logger.close()  # Important!

    3. Auto-managed (via World):
    >>> world = World.with_logging("my_sim")
    >>> world.run(solver, duration, dt)
    >>> # Logger automatically managed

    Examples
    --------
    >>> # Log only positions and velocities
    >>> logger = CSVLogger("minimal.csv", fields=["p", "v"])
    >>>
    >>> # Large buffer for long simulations
    >>> logger = CSVLogger("long_sim.csv", buffer_size=5000)
    """

    def __init__(
        self,
        filepath: str | Path,
        buffer_size: int = 1000,
        fields: list[str] | None = None
    ) -> None:
        self.filepath = Path(filepath)
        self.buffer_size = buffer_size
        self.fields = fields if fields is not None else ["p", "q", "v", "w", "f", "tau"]

        # Validate fields
        valid_fields = {"p", "q", "v", "w", "f", "tau"}
        invalid = set(self.fields) - valid_fields
        if invalid:
            raise ValueError(
                f"Invalid fields: {invalid}. Valid options: {valid_fields}"
            )

        self._buffer: list[list[str]] = []
        self._file: TextIO | None = None
        self._writer: Any = None  # csv.writer is a function, not a type
        self._header_written = False

        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> CSVLogger:
        """Open file for writing."""
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file, flushing any remaining data."""
        self.close()

    def _get_val(self, b: RigidBody6DOF, field: str) -> Any:
        """Retrieve field value from body."""
        if hasattr(b, field):
            return getattr(b, field)

        # Special handling for force/torque (accumulated values)
        if field in {'f', 'tau'}:
            gf = b.generalized_force()
            if field == 'f':
                return gf[:3]
            if field == 'tau':
                return gf[3:]

        return None

    def _write_header(self, world: Any) -> None:
        """Generate and write CSV header row."""
        hdr = ["t"]

        # Map fields to component suffixes
        field_components = {
            "p": ["x", "y", "z"],
            "q": ["x", "y", "z", "w"],
            "v": ["x", "y", "z"],
            "w": ["x", "y", "z"],
            "f": ["x", "y", "z"],
            "tau": ["x", "y", "z"],
        }

        for b in world.bodies:
            for field in self.fields:
                if field in field_components:
                    for component in field_components[field]:
                        hdr.append(f"{b.name}.{field}_{component}")
                else:
                    # Fallback for unknown fields
                    val = self._get_val(b, field)
                    if val is not None and hasattr(val, '__len__'):
                        for i in range(len(val)):
                            hdr.append(f"{b.name}.{field}_{i}")
                    else:
                        hdr.append(f"{b.name}.{field}")

        if self._writer:
            self._writer.writerow(hdr)
            if self._file:
                self._file.flush()  # Ensure header written immediately

        self._header_written = True

    def log(self, world: Any) -> None:
        """
        Log current world state to buffer.

        Parameters
        ----------
        world : World
            World object to log

        Notes
        -----
        Automatically opens file on first call if not using context manager.
        Writes to disk when buffer is full.
        """
        # Auto-open if not in context manager
        if self._file is None:
            self.__enter__()

        if not self._header_written:
            self._write_header(world)

        # Build row
        row = [f"{world.t:.10f}"]  # High precision time
        for b in world.bodies:
            for field in self.fields:
                val = self._get_val(b, field)
                if val is not None:
                    if hasattr(val, '__iter__'):
                        # Vector/array field
                        row.extend(f"{v:.10e}" for v in val)  # Scientific notation
                    else:
                        # Scalar field
                        row.append(f"{val:.10e}")

        self._buffer.append(row)

        # Flush if buffer full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered data to disk and clear buffer."""
        if self._writer and self._buffer:
            self._writer.writerows(self._buffer)
            if self._file:
                self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        """Flush remaining data and close file."""
        self.flush()
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

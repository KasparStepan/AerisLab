from __future__ import annotations
import csv
import os
from typing import List, Optional, TextIO, Any
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF

class CSVLogger:
    """
    Logs simulation state to a CSV file.
    
    Optimized for performance:
    1. Keeps file handle open (avoids repeated open/close syscalls).
    2. Buffers data in memory and writes in chunks.
    3. Implements Context Manager protocol for safe resource handling.
    """
    def __init__(self, filepath: str, buffer_size: int = 1000, fields: Optional[List[str]] = None) -> None:
        self.filepath = filepath
        self.buffer_size = buffer_size
        self.fields = fields if fields is not None else ["p", "q", "v", "w", "f", "tau"]
        
        self._buffer: List[List[float]] = []
        self._file: Optional[TextIO] = None
        self._writer = None
        self._header_written = False
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def __enter__(self) -> CSVLogger:
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get_val(self, b: Any, field: str) -> Any:
        """Retrieve value from body, handling both attributes and generalized forces."""
        if hasattr(b, field):
            return getattr(b, field)
        
        # Fallback for forces on RigidBody6DOF which might not be stored as attributes
        if field in ['f', 'tau'] and hasattr(b, 'generalized_force'):
            gf = b.generalized_force()
            if field == 'f': return gf[:3]
            if field == 'tau': return gf[3:]
        return None

    def _write_header(self, world: Any) -> None:
        hdr = ["t"]
        
        # Map of fields to their component suffixes
        vector_fields = {
            "p": ["x", "y", "z"],
            "q": ["x", "y", "z", "w"],
            "v": ["x", "y", "z"],
            "w": ["x", "y", "z"],
            "f": ["x", "y", "z"],
            "tau": ["x", "y", "z"],
        }
        
        for b in world.bodies:
            base = f"{b.name}"
            for field in self.fields:
                if field in vector_fields:
                    for s in vector_fields[field]:
                        hdr.append(f"{base}.{field}_{s}")
                else:
                    # Fallback for scalar or unknown fields
                    val = self._get_val(b, field)
                    if val is not None and hasattr(val, '__len__') and not isinstance(val, str):
                        for i in range(len(val)):
                            hdr.append(f"{base}.{field}_{i}")
                    else:
                        hdr.append(f"{base}.{field}")
            
        if self._writer:
            self._writer.writerow(hdr)
            self._file.flush()  # Ensure header is written to disk immediately
        self._header_written = True

    def log(self, world: Any) -> None:
        # Auto-open if not used in context manager (backward compatibility)
        if self._file is None:
            self.__enter__()

        if not self._header_written:
            self._write_header(world)
            
        row = [str(world.t)]
        for b in world.bodies:
            for field in self.fields:
                val = self._get_val(b, field)
                if val is not None:
                    if hasattr(val, '__iter__'):
                        row.extend(map(str, val))
                    else:
                        row.append(str(val))
            
        self._buffer.append(row)
        
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._writer and self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

from __future__ import annotations
import csv
import os
from typing import Optional
import numpy as np
from aerislab.dynamics.body import RigidBody6DOF

Array = np.ndarray

class CSVLogger:
    """
    Append time-stamped rows per body: p, q, v, w, resultant f, tau.
    Creates header once. Safe for repeated runs.
    """
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self._header_written = False
        # ensure directory
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def _ensure_header(self, world) -> None:
        if self._header_written:
            return
        hdr = ["t"]
        for b in world.bodies:
            base = f"{b.name}"
            hdr += [f"{base}.p_x", f"{base}.p_y", f"{base}.p_z",
                    f"{base}.q_x", f"{base}.q_y", f"{base}.q_z", f"{base}.q_w",
                    f"{base}.v_x", f"{base}.v_y", f"{base}.v_z",
                    f"{base}.w_x", f"{base}.w_y", f"{base}.w_z",
                    f"{base}.F_x", f"{base}.F_y", f"{base}.F_z",
                    f"{base}.T_x", f"{base}.T_y", f"{base}.T_z"]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(hdr)
        self._header_written = True

    def log(self, world) -> None:
        self._ensure_header(world)
        row = [world.t]
        for b in world.bodies:
            row += [*b.p, *b.q, *b.v, *b.w, *b.f, *b.tau]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)

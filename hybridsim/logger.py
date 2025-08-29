from __future__ import annotations
import csv
from pathlib import Path

class CSVLogger:
    """Minimal CSV logger for body states and resultant forces/torques."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fp = self.path.open("w", newline="")
        self._w = csv.writer(self._fp)
        self._w.writerow(self._header_written_example())
        self._fp.flush()

    def _header_written_example(self):
        cols = ["time"]
        # We don't know bodies yet; write flexible header instructions
        cols.append("NOTE: Columns repeat per body: name,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,w_x,w_y,w_z,F_x,F_y,F_z,T_x,T_y,T_z")
        return cols

    def write(self, t: float, world) -> None:
        row = [f"{t:.9f}"]
        for b in world.bodies:
            row.extend([
                b.name, f"{b.p[0]:.9e}", f"{b.p[1]:.9e}", f"{b.p[2]:.9e}",
                f"{b.q[0]:.9e}", f"{b.q[1]:.9e}", f"{b.q[2]:.9e}", f"{b.q[3]:.9e}",
                f"{b.v[0]:.9e}", f"{b.v[1]:.9e}", f"{b.v[2]:.9e}",
                f"{b.w[0]:.9e}", f"{b.w[1]:.9e}", f"{b.w[2]:.9e}",
                f"{b.F[0]:.9e}", f"{b.F[1]:.9e}", f"{b.F[2]:.9e}",
                f"{b.T[0]:.9e}", f"{b.T[1]:.9e}", f"{b.T[2]:.9e}",
            ])
        self._w.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Iterable

@dataclass
class CSVLogger:
    """Very small CSV logger; avoids heavy deps so tests run light-weight."""
    path: str
    write_header: bool = True
    _header_written: bool = False

    def log(self, time: float, bodies: Iterable["RigidBody6DOF"]) -> None:
        rows = []
        for b in bodies:
            rows.append({
                "time": time,
                "body": b.name,
                "px": b.position[0], "py": b.position[1], "pz": b.position[2],
                "qx": b.orientation[0], "qy": b.orientation[1],
                "qz": b.orientation[2], "qw": b.orientation[3],
                "vx": b.linear_velocity[0], "vy": b.linear_velocity[1], "vz": b.linear_velocity[2],
                "wx": b.angular_velocity[0], "wy": b.angular_velocity[1], "wz": b.angular_velocity[2],
                "fx": b.force[0], "fy": b.force[1], "fz": b.force[2],
                "tx": b.torque[0], "ty": b.torque[1], "tz": b.torque[2],
            })
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if self.write_header and not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(rows)

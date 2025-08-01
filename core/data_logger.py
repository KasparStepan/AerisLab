import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class Logger:
    def __init__(self, log_to_file=False, filename="simulation_log.csv"):
        self.log_to_file = log_to_file
        self.filename = filename
        # Initialize empty DataFrame with columns for all properties
        self.logs = pd.DataFrame(columns=[
            "Timestamp", "Time", "Body",
            "Pos_X", "Pos_Y", "Pos_Z",
            "Quat_X", "Quat_Y", "Quat_Z", "Quat_W",
            "Vel_X", "Vel_Y", "Vel_Z",
            "AngVel_X", "AngVel_Y", "AngVel_Z",
            "Force_X", "Force_Y", "Force_Z",
            "Torque_X", "Torque_Y", "Torque_Z"
        ])
        if self.log_to_file:
            # Write empty CSV with headers
            self.logs.to_csv(self.filename, index=False)

    def log_bodies(self, time, bodies):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        for body in bodies:
            # Create log entry as a dictionary
            log_entry = {
                "Timestamp": timestamp,
                "Time": time,
                "Body": body.name,
                "Pos_X": body.position[0],
                "Pos_Y": body.position[1],
                "Pos_Z": body.position[2],
                "Quat_X": body.orientation[0],
                "Quat_Y": body.orientation[1],
                "Quat_Z": body.orientation[2],
                "Quat_W": body.orientation[3],
                "Vel_X": body.linear_velocity[0],
                "Vel_Y": body.linear_velocity[1],
                "Vel_Z": body.linear_velocity[2],
                "AngVel_X": body.angular_velocity[0],
                "AngVel_Y": body.angular_velocity[1],
                "AngVel_Z": body.angular_velocity[2],
                "Force_X": body.force[0],
                "Force_Y": body.force[1],
                "Force_Z": body.force[2],
                "Torque_X": body.torque[0],
                "Torque_Y": body.torque[1],
                "Torque_Z": body.torque[2]
            }
            # Append to DataFrame
            self.logs = pd.concat([self.logs, pd.DataFrame([log_entry])], ignore_index=True)
            # Console output
            print(f"[LOG] Time={time:.4f}, Body={body.name}, "
                  f"Pos=[{body.position[0]:.6f}, {body.position[1]:.6f}, {body.position[2]:.6f}], "
                  f"Quat=[{body.orientation[0]:.6f}, {body.orientation[1]:.6f}, {body.orientation[2]:.6f}, {body.orientation[3]:.6f}], "
                  f"Vel=[{body.linear_velocity[0]:.6f}, {body.linear_velocity[1]:.6f}, {body.linear_velocity[2]:.6f}], "
                  f"AngVel=[{body.angular_velocity[0]:.6f}, {body.angular_velocity[1]:.6f}, {body.angular_velocity[2]:.6f}], "
                  f"Force=[{body.force[0]:.6f}, {body.force[1]:.6f}, {body.force[2]:.6f}], "
                  f"Torque=[{body.torque[0]:.6f}, {body.torque[1]:.6f}, {body.torque[2]:.6f}]")
            # Append to CSV if enabled
            if self.log_to_file:
                pd.DataFrame([log_entry]).to_csv(self.filename, mode='a', header=False, index=False)

    def print_all_logs(self):
        if self.logs.empty:
            print("[LOGGER] No logs to print")
            return
        print("[LOGGER] Printing all logs:")
        for _, log in self.logs.iterrows():
            print(f"[LOG] Time={log['Time']:.4f}, Body={log['Body']}, "
                  f"Pos=[{log['Pos_X']:.6f}, {log['Pos_Y']:.6f}, {log['Pos_Z']:.6f}], "
                  f"Quat=[{log['Quat_X']:.6f}, {log['Quat_Y']:.6f}, {log['Quat_Z']:.6f}, {log['Quat_W']:.6f}], "
                  f"Vel=[{log['Vel_X']:.6f}, {log['Vel_Y']:.6f}, {log['Vel_Z']:.6f}], "
                  f"AngVel=[{log['AngVel_X']:.6f}, {log['AngVel_Y']:.6f}, {log['AngVel_Z']:.6f}], "
                  f"Force=[{log['Force_X']:.6f}, {log['Force_Y']:.6f}, {log['Force_Z']:.6f}], "
                  f"Torque=[{log['Torque_X']:.6f}, {log['Torque_Y']:.6f}, {log['Torque_Z']:.6f}]")

    def save_to_csv(self, filename=None):
        if self.logs.empty:
            print("[LOGGER] No data to save")
            return
        filename = filename or self.filename
        self.logs.to_csv(filename, index=False)
        print(f"[LOGGER] Saved logs to {filename}")

    def plot_property(self, property_name, body_name, title=None, ylabel=None):
        if self.logs.empty:
            print("[LOGGER] No data to plot")
            return
        # Filter logs for the specified body
        body_logs = self.logs[self.logs['Body'] == body_name]
        if body_logs.empty:
            print(f"[LOGGER] No logs for body {body_name}")
            return
        plt.figure()
        plt.plot(body_logs['Time'], body_logs[property_name])
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel or property_name)
        plt.title(title or f"{property_name} vs Time for {body_name}")
        plt.grid(True)
        plt.show()
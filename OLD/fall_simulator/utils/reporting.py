import img2pdf
import os

image_files = [
    #"outputs/trajectory_payload.png",
    #"outputs/trajectory_parachute.png",
    "outputs/position_vs_time.png",
    "outputs/velocity_vs_time.png",
    "outputs/acceleration_vs_time.png",
    "outputs/energy_vs_time.png",
]

def create_pdf_report(output_pdf_path):
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_files))






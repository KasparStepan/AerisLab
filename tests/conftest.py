import os
import sys

# Get the path to the project root (one level up from 'tests')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')

# Add 'src' to sys.path
sys.path.insert(0, src_path)

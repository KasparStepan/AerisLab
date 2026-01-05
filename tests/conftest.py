import sys
import os

# Add the project root directory to sys.path to allow importing AerisLab
# This ensures tests can run even if the package is not installed in the environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

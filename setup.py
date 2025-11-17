from setuptools import setup, find_packages
import os

print("\n\nWARNING: This library for ORCA is still buggy and not very reliable. Use at your own risk!\n\n")

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    jorca_requirements = f.read().splitlines()

setup(
    name='jorca',
    version='0.0.1',
    packages= find_packages(),
    install_requires = jorca_requirements
)
import os
from os import path
import warnings

from setuptools import find_namespace_packages, setup

import fasterrl

ROOT = path.abspath(path.dirname(__file__))

# Get requirements from file
with open(path.join(ROOT, "requirements.txt")) as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

# Get the long description from the README file
with open(path.join(ROOT, "README.md")) as f:
    readme = f.read()

# Set path to save local files
if "FASTERRL_LOGDIR" in os.environ:
    local_path = os.environ["FASTERRL_LOGDIR"]
else:
    local_path = path.join(ROOT, "local")
    # Append environment variable definition to .bashrc
    with open(path.expanduser("~/.bashrc"), "a") as outfile:
        outfile.write("\n")
        outfile.write(f"export FASTERRL_LOGDIR={local_path}")
    os.environ['FASTERRL_LOGDIR'] = local_path

# Create local directory
try:
    os.mkdir(local_path)
except:
    warnings.warn(f"Folder {local_path} already exists")

# Create additional directories
for folder in ["logs", "results", "runs", "weights"]:
    folder_path = path.join(local_path,folder)
    try:
        os.mkdir(folder_path)
    except:
        warnings.warn(f"Folder {folder_path} already exists")

setup(
    name="fasterrl",
    author="Lucas Souza",
    author_email="None",
    license="MIT",
    platforms=["any"],
    url="https://github.com/lucasosouza/fasterrl",
    description="Library created for RL Experience Sharing experiments",
    long_description=readme,
    long_description_content_type="text/markdown",
    version=0.1,
    install_requires=requirements,
    python_requires=">=3.7, <4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["fasterrl"],
    project_urls={
        "Bug Reports": "https://github.com/lucasosouza/fasterrl/issues",
        "Source": "https://github.com/lucasosouza/fasterrl",
    },
    test_suite="tests",
    tests_require=["pytest>=4.4.0"],
)
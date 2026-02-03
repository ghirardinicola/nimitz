#!/usr/bin/env python3
"""
NIMITZ - Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = []
req_file = Path(__file__).parent / "requirements.txt"
if req_file.exists():
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            # Skip git URLs and comments
            if line and not line.startswith("#") and not line.startswith("git+"):
                requirements.append(line)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="nimitz",
    version="0.2.0",
    description="Transform images into collectible cards with quantified statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NIMITZ Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["cli", "core", "embed", "cluster", "viz", "image_card", "main"],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nimitz=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

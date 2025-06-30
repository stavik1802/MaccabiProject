#!/usr/bin/env python3
"""
Setup script for Maccabi Football GPS Tracking Analysis Project

This project processes GPS tracking data from football matches to analyze player performance,
detect match periods, analyze substitutions, and provide comprehensive performance metrics.
"""

from setuptools import setup, find_packages
import os

# Read the README file if it exists
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "Maccabi Football GPS Tracking Analysis Project"

# Read requirements from requirements.txt
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="maccabi-gps-analysis",
    version="1.0.0",
    author="Stav Karasik",
    author_email="stavos114@gmail.com",
    description="Comprehensive football GPS tracking data analysis and machine learning pipeline",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/maccabi-gps-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Sports Analytics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sports",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "maccabi-preprocess=pre_proccess:main",
            "maccabi-train=train:main",
            "maccabi-predict=prediction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv", "*.xlsx"],
    },
    keywords="football, soccer, GPS, tracking, sports analytics, machine learning, performance analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/maccabi-gps-analysis/issues",
        "Source": "https://github.com/yourusername/maccabi-gps-analysis",
        "Documentation": "https://github.com/yourusername/maccabi-gps-analysis/blob/main/README.md",
    },
) 
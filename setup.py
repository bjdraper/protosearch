from setuptools import setup, find_packages

setup(
    name="acyltransferase",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "scikit-learn", "biopython",
        "pyyaml", "matplotlib", "seaborn",
    ],
)


from setuptools import find_packages, setup

setup(

    name="resource-recommendation",

    version="0.1.0",

    packages=find_packages(),

    install_requires=[

        "pandas",

        "numpy",

        "scikit-learn",

        "matplotlib",

        "seaborn",

        "fastapi",

        "uvicorn",

        "pytest",

        "jupyter"

    ],

)


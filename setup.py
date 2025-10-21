from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="resource-recommendation",
    version="0.1.0",
    author="Iyanujesu Akinyefa",
    author_email="iyanujesuakinyefa@gmail.com",
    description="AI-powered resource recommendation engine for professional services teams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tee-works/AI-resource-recommendation-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "fastapi>=0.97.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.9",
        "pulp>=2.7.0",
        "pytest>=7.3.1",
        "jupyter>=1.0.0",
        "notebook>=6.5.4",
    ],
)

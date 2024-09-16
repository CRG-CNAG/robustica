from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="robustica",
    version="0.1.4",
    packages=["robustica"],
    python_requires=">=3.8",
    package_data={"": ["LICENSE", "*.md","*.ipynb","*.yml"]},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "tqdm",
    ],
    extras_require={
        "extra": ["scikit-learn-extra"]
    },
    author="Miquel Anglada Girotto",
    author_email="miquel.anglada@crg.eu",
    description="Fully cumstomizable robust Independent Components Analysis (ICA)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/CRG-CNAG/robustica",
    project_urls={
        "Issues": "https://github.com/CRG-CNAG/robustica/issues",
        "Documentation": "https://crg-cnag.github.io/robustica/"
    },
)

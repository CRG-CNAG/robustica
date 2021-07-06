from setuptools import setup

setup(
    name="robustica",
    version="0.1.0",
    packages=["robustica"],
    python_requires=">=3.8",
    package_data={"": ["LICENSE", "*.md","*.ipynb","*.yml"]},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-learn-extra",
        "joblib",
        "tqdm",
    ],
    author="Miquel Anglada Girotto",
    author_email="miquelangladagirotto@gmail.com",
    description="Fully cumstomizable robust Independent Components Analysis (ICA)",
    url="https://github.com/MiqG/robustica",
    project_urls={"Issues": "https://github.com/MiqG/robustica/issues"},
)

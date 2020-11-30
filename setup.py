# Imports: third party
from setuptools import setup, find_packages

setup(
    name="ml4c3",
    version="0.1",
    description="Machine Learning for Cardiology and Critical Care package",
    url="https://github.com/aguirre-lab/ml4c3",
    python_requires=">=3.6",
    install_requires=[],
    packages=find_packages(),
    package_data={
        "ml4c3": ["visualizer/*", "visualizer/assets/*"],
        "ingest": ["edw/queries-pipeline/*"],
    },
    include_package_data=True,
)

from setuptools import setup, find_packages

setup(
    name="clustersense_package",
    version="0.0.1",
    author="Raj Kiran Reddy",
    description="ClusterSense AI — Customer Segmentation ML Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "streamlit",
    ]
)
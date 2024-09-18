from setuptools import setup, find_packages

setup(
    name="metalatte",
    version="0.1",
    author="yinuo",
    author_email="yzhang@u.duke.nus.edu",
    description="MetaLATTE model for metal-binding protein analysis",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.1+cu117",
        "numpy==1.22.4",
        "lightning==2.1.2",
        "transformers>=4.38.2",
    ],
    python_requires='>=3.8',
)
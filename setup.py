import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ensemble-free-data-assimilation",
    version="0.0.1",
    author="Yuuichi Asahi",
    author_email="y.asahi@nr.titech.ac.jp",
    description="Ensemble Free Data Assimilation Model - Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasahi-hpc/efda_prototype",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'einops',
        'dask',
        'xarray[io]',
        'xarray[viz]',
        'torch',
        'torchvision',
        'gymnasium',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

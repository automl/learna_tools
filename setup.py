#create a setup.py file to install the package
from setuptools import setup, find_packages


setup(
    name="learna_tools",
    version="0.1.0",
    description="RNA Design with automated reinforcement learning.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "readme-renderer==16.0",
        "Distance==0.1.3",
        "GraKeL==0.1.8",
        "numpy==1.9.5",
        "pandas==1.1.5",
        "pyaml==23.5.8",
        "pynisher==0.6.4",
        "pytest==7.0.1",
        "pylint==2.13.9",
        "psutil==5.9.8",
        "configspace==0.4.19",
        "scipy==1.5.4",
        "Sphinx==5.3.0",
        "tensorflow==1.4.0",
        "scikits.bootstrap==1.1.0",
        "hpbandster==0.7.4",
        "tqdm==4.64.1",
        "black==22.8.0",
        "tensorforce==0.3.3",
        "dataclasses==0.8",
        "munch==4.0.0",
        "pyyaml==6.0.1",
        "scikit-learn==0.24.2",
        "LinearFold==1.1",
        "tabulate==0.8.10",
        "matplotlib==3.3.4",
        "biopython==1.79",
        "logomaker==0.8",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        # Add other relevant classifiers.
    ],
    scripts=[
        'bin/learna',
        'bin/meta-learna',
        'bin/meta-learna-adapt',
        'bin/liblearna',
    ],
# Optional
    author='Frederic Runge',
    author_email='runget@cs.uni-freiburg.de',
    license='MIT',
    keywords='RNA Design',
    url='https://github.com/automl/learna_tools.git'
)

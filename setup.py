from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Taxidemand",
    version="0.1.0",
    author="mahmood abdali",
    packages=find_packages(),
    install_requires=requirements,
)

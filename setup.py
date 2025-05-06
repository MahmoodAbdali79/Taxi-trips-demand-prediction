from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tap40",
    version="0.1.0",
    author="mahmood abdali",
    packages=find_packages(),
    # packages=find_packages(where="src"),
    # package_dir={"":"src"},
    install_requires=requirements,
)

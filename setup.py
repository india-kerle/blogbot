from pathlib import Path
from setuptools import find_packages
from setuptools import setup


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent 

setup(
    name="blogbot",
    long_description=open(BASE_DIR / "README.md").read(),
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    additional_requires=read_lines(BASE_DIR / "requirements_dev.txt"),
    packages=find_packages(exclude=["data", "tests"]),
    version="0.0.0",
    description="Train a model to identify PII in blog posts",
    author="india-kerle",
    license="proprietary",
)
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='paritybench',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ntent_type="text/markdown",
    url="https://github.com/jansel/pytorch-jit-paritybench",
    packages = ["paritybench"],
    install_requires=required)

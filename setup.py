from pathlib import Path
from setuptools import setup, find_packages

description = ['Building a toy dump truck that can follow lines drawn in chalk']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='chalk',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=dependencies,
    author='Tim Fanselow, Rahul Mishra',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/chalktracks/chalktracks',
)
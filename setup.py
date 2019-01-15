import os
from distutils.core import setup

current_file_path = os.path.abspath(os.path.dirname(__file__))

readme_file_path = os.path.join(current_file_path, 'README.md')
with open(readme_file_path, 'r') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pytobop',
    version='0.1',
    packages=['pytobop'],
    url='',
    license='MIT',
    author='zio',
    author_email='',
    description='PyTorch Boilerplate',
    install_requires=requirements,
    python_requires=">=3.5",
    long_description=readme,
    long_description_content_type='text/markdown',
)

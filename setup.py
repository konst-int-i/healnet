from setuptools import setup, find_packages

setup(
    name='healnet',
    version='0.0.1',
    package_dir={'': 'healnet'},
    packages=find_packages(),
    install_requires=[
        'einops',
        'torch',
        'torchvision',
        'openslide-python',
        'seaborn',
        'pandas'
    ]
)
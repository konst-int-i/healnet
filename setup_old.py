from setuptools import setup, find_packages

setup(
    name='healnet',
    version='0.0.1',
    packages=['healnet.models', 'healnet.baselines', 'healnet.etl', 'healnet.utils'],
    install_requires=[
        'einops',
        'torch',
        'torchvision',
        'openslide-python',
        'seaborn',
        'pandas'
    ]
)
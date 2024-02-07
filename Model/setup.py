from setuptools import setup, find_packages

setup(
    name='ML-beria',
    version='0.1.0',
    description='Xbox recommended system model',
    author='Beria C. KALPELBE',
    author_email='beria@aims.ac.za',
    url='',
    packages=find_packages(), 
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'math',
        'tqdm'
    ],
)

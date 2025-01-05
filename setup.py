from setuptools import setup, find_packages

setup(
    name="sleep_eeg",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'mne>=1.5.1',
        'numpy>=1.24.0',
        'tensorflow>=2.13.0',
        'flask>=2.0.0',
        'flask-cors>=4.0.0',
        'matplotlib>=3.7.0',
        'scipy>=1.9.0',
        'scikit-learn>=1.0.0'
    ]
) 
from setuptools import setup, find_packages

setup(
    name="SecondContact",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests',                   # For handling HTTP requests
        'torch',                      # PyTorch for machine learning
        'transformers==4.45.0',       # Hugging Face Transformers library
        # 'accelerate==0.34.1',         # For optimized training and inference
    ],
    entry_points={
        'console_scripts': [
            # Add your script entry points here if needed
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="rankdiff",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "matplotlib>=3.3.2",
        "design-bench>=2.0.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "seaborn>=0.11.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="RankDiff: Rank-Based Diffusion Models for Offline Black-Box Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RankDiff",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 
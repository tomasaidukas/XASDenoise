from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Build-time toggles (full by default)
# Full installation with examples and pytorch (default):
#     pip install .
# To avoid installing pytorch: 
#     XASDENOISE_WITH_ML=0 pip install .
# To avoid installing example scripts and data:
#     XASDENOISE_WITH_EXAMPLES=0 pip install .
# Exclude both:
#     XASDENOISE_WITH_ML=0 XASDENOISE_WITH_EXAMPLES=0 pip install .
WITH_TORCH = os.getenv("XASDENOISE_WITH_ML", "1") == "1"
WITH_EXAMPLES = os.getenv("XASDENOISE_WITH_EXAMPLES", "1") == "1"

install_requires = read_requirements()
if not WITH_TORCH:
    # drop torch/gpytorch if they are in requirements.txt
    install_requires = [req for req in install_requires if not req.split("==")[0].lower().startswith(("torch", "gpytorch"))]

# Exclude examples package when requested
exclude_pkgs = ["xasdenoise.examples*"] if not WITH_EXAMPLES else []

setup(
    name="XASDenoise",
    version="0.1.0",
    description="X-ray Absorption Spectroscopy Denoising Package",
    author="Tomas Aidukas",
    author_email="tomasaiduk@gmail.com",
    packages=find_packages(include=["xasdenoise", "xasdenoise.*"], exclude=exclude_pkgs),
    install_requires=install_requires,
    include_package_data=WITH_EXAMPLES,
    python_requires=">=3.11",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)

# XASDenoise

**XASDenoise** is a Python package for denoising and preprocessing **X-ray Absorption Spectroscopy (XAS)** data.  
It implements a data preprocessing method called **stationarity warping**, which aims at mitigating signal non-stationarity.
The package also includes advanced denoising techniques based on **Gaussian Processes** and **convolutional autoencoders**. The package comes with ready-to-run Jupyter notebook examples and example XAS datasets.

---

## Features
- **Gaussian Process**-based denoising for probabilistic signal recovery
- **Convolutional autoencoder** training and denoising functionality for non-linear noise removal
- **Stationarity warping** to mitigate signal non-stationarity and improve performance of any denoising method
- Preprocessing utilities for XAS data
- Example datasets and Jupyter notebooks included

---

## Installation
```
pip install git+https://github.com/tomasaidukas/XASDenoise.git
```

or

```
git clone https://github.com/tomasaidukas/XASDenoise.git
cd XASDenoise
pip install -e .
```

## Examples & Data

The repository includes:

* **`examples/`** — Jupyter notebooks:

  * `example0_spectrum_object_and_preprocessing.ipynb` — creating and processing spectrum objects
  * `example1_regular_denoiser.ipynb` — basic denoising
  * `example2_gaussian_process_denoiser.ipynb` — Gaussian Process denoising
  * `example3_encoder_denoiser.ipynb` — autoencoder denoising
  * `example4_time_resolved_data_denoising.ipynb` — time-resolved spectra denoising
  * `example5_time_resolved_data_noise2noise.ipynb` — time-resolved spectra denoising using a noise2noise convolutional autoencoder
* **`examples/data/`** — example XAS datasets

--- 

## Requirements

* Python >= 3.12
* See [`requirements.txt`](requirements.txt) for core dependencies

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
# PyTorch Implementation of Context-Encoder

This is a [PyTorch](https://pytorch.org/) re-implementation of *Context Encoder* as proposed by Pathak et al. in "Context encoders: Feature learning by inpainting". The code is referenced from the [original github repository](https://github.com/pathak22/context-encoder) and [PyTorch-GAN library](https://github.com/eriklindernoren/PyTorch-GAN).

The entire codebase has been written from scatch. The core training framework can be found in `<path-to-Context-Encoder-directory>\src\main.py`. The generator and adversarial networks are configured within `<path-to-Context-Encoder-directory>\src\networks` directory. 

The implementation is also extended to use style loss, content loss and total variation loss using a pre-trained VGG19 model inspire by *Context-Aware Semantic In-painting*. However, the extension could not be investigated owing to resource constraints.

## Installation
This code is written in `Python 3.10.4` and requires the packages listed in `requirements.yml` (for cpu-based execution) or `requirements_gpu.yml` (for using gpu).

To run the code, set up a virtual environment using `conda`:

```
cd <path-to-Context-Encoder-directory>
conda env create --file requirements_gpu.yml
conda activate aip_proj
```

## Running experiments

Experiments have been executed on Paris Street View and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. Paris Street View dataset is not publicly available. Send an email to prof. Deepak Pathak (dpathak@cs.cmu.edu) to acquire the dataset. Download the datasets and extract the contents into the `<path-to-Context-Encoder-directory>/data/` directory.

All the experimental configurations can be accessed from `<path-to-Context-Encoder-directory>\src\config.json`. To run an experiment 

```
cd <path-to-Context-Encoder-directory>

# activate virtual environment
conda activate aip_proj

# change to source directory
cd src

# run experiment
python main.py
```

## Examples

### Paris Street View

![Paris Street View](outputs/Results_parisStreetView.png?raw=true "Paris Street View")

### CelebA

![CelebA](outputs/Results_celeba.png?raw=true "CelebA")

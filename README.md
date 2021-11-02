# Neural Nano-Optics for High-quality Thin Lens Imaging
### [Project Page]() | [Paper]() | [Data](https://drive.google.com/drive/folders/1fsAvN9MPtN5jJPeIFjWuLUY9Hp8NNkar?usp=sharing)

[Ethan Tseng](https://ethan-tseng.github.io), [Shane Colburn](https://scholar.google.com/citations?user=WLnx6NkAAAAJ&hl=en), [James Whitehead](https://scholar.google.com/citations?user=Hpcg0h4AAAAJ&hl=en), [Luocheng Huang](https://scholar.google.com/citations?user=x9UDJHgAAAAJ&hl=en), [Seung-Hwan Baek](https://sites.google.com/view/shbaek/), [Arka Majumdar](https://scholar.google.com/citations?user=DpIGlW4AAAAJ&hl=en), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

This code implements a differentiable proxy model for simulating meta-optics and a neural feature propagation deconvolution method. These components are optimized end-to-end using machine learning optimizers.

The experimental results from the manuscript and the supplemental information are reproducible with this implementation. The proposed differentiable proxy model, neural feature propagation, and end-to-end optimization framework are implemented completely in TensorFlow, without dependency on third-party libraries. While the fastest inference and training times are achieved by running on state-of-the-art GPUs, this code can be run solely with standard TensorFlow packages, ensuring high portability and ease of reproducibility.


## Training
To perform end-to-end training (of meta-optic and deconvolution) execute the 'run_train.sh' script. The model checkpoint which includes saved parameters for both the meta-optic and deconvolution will be saved to 'training/ckpt'. The folder 'training/data' contains a subset of the training and test data that we used for optimizing our end-to-end imaging pipeline.


## Testing
To perform inference on real-world captures launch the "test.ipynb" notebook in Jupyter Notebook and step through the cells. The notebook will load in a finetuned checkpoint of our neural feature propagation network from 'experimental/ckpt' which will process captured sensor measurements located in 'experimental/data'. The reconstructed images will be displayed within the notebook.


## Requirements
This code has been tested with Python 3.6.10 using TensorFlow 2.2.0 running on Linux with an Nvidia P100 GPU with 16GB RAM.

We installed the following library packages to run this code:
```
TensorFlow >= 2.2
TensorFlow Probability
TensorFlow Addons
Numpy
Scipy
matplotlib
jupyter-notebook
```

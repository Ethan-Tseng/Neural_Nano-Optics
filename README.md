# Neural Nano-Optics for High-quality Thin Lens Imaging
### [Project Page]() | [Paper]() | [Data](https://drive.google.com/drive/folders/1fsAvN9MPtN5jJPeIFjWuLUY9Hp8NNkar?usp=sharing)

[Ethan Tseng](https://ethan-tseng.github.io), [Shane Colburn](https://scholar.google.com/citations?user=WLnx6NkAAAAJ&hl=en), [James Whitehead](https://scholar.google.com/citations?user=Hpcg0h4AAAAJ&hl=en), [Luocheng Huang](https://scholar.google.com/citations?user=x9UDJHgAAAAJ&hl=en), [Seung-Hwan Baek](https://sites.google.com/view/shbaek/), [Arka Majumdar](https://scholar.google.com/citations?user=DpIGlW4AAAAJ&hl=en), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

This code implements a differentiable proxy model for simulating meta-optics and a neural feature propagation deconvolution method. These components are optimized end-to-end using machine learning optimizers.

The experimental results from the manuscript and the supplemental information are reproducible with this implementation. The proposed differentiable proxy model, neural feature propagation, and end-to-end optimization framework are implemented completely in TensorFlow, without dependency on third-party libraries.

## Training
To perform end-to-end training (of meta-optic and deconvolution) execute the 'run_train.sh' script. The model checkpoint which includes saved parameters for both the meta-optic and deconvolution will be saved to 'training/ckpt'. The folder 'training/data' contains a subset of the training and test data that we used for optimizing our end-to-end imaging pipeline.

## Testing
To perform inference on real-world captures launch the "test.ipynb" notebook in Jupyter Notebook and step through the cells. The notebook will load in a finetuned checkpoint of our neural feature propagation network from 'experimental/ckpt' which will process captured sensor measurements located in 'experimental/data'. The reconstructed images will be displayed within the notebook.

Additional captured sensor measurements can be found in the [data repository](https://drive.google.com/drive/folders/1fsAvN9MPtN5jJPeIFjWuLUY9Hp8NNkar?usp=sharing).

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

## Citation
If you find our work useful in your research, please cite:
```
@article{Tseng2021NeuralNanoOptics,
    title   = "Neural Nano-Optics for High-quality Thin Lens Imaging",
    author  = "Tseng, Ethan and Colburn, Shane and Whitehead, James and Huang, Luocheng
               and Baek, Seung-Hwan and Majumdar, Arka and Heide, Felix",
    journal = "Nature Communications",
    volume  = ,
    number  = ,
    pages   = ,
    year    = 2021
}
```

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. The training data in the folder 'training/data' comes from the [INRIA Holidays Dataset](https://lear.inrialpes.fr/~jegou/data.php).

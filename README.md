# train-clustered

This repository contains the code required to train a set of drug-target inhibiton classification models for various transporter proteins. These models are currently used internally at the [Pharmacoinformatics Research Group](https://pharminfo.univie.ac.at/) of the University of Vienna.

## Usage

We provide the full code required to train these models in the form of a Jupyter/IPython Notebook. It is recommended to use [conda](https://docs.conda.io/en/latest/) or any compatible tool to install the required dependencies.

``` shell
conda create -f train-clustered.yml
conda activate train-clustered
jupyter lab
```

For our own convenience, the code is configured for training on the Vienna [Life Science Compute Cluster (LiSC)](https://lisc.univie.ac.at/). If you want to run it for yourself locally or on your own cluster, please modify the in the section "Configure task runner" of the provided Notebook.

## Limitations

In this repository we only provide dataset of drug-target interactions extracted from the public ChEMBL database. For our internal models we use a larger dataset, parts of which are confidential.

## License

TODO

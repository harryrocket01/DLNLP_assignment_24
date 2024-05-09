# README

# DLNLP_assignment23_24

## Description

This codebase is an exploration into the rabbit hole of spelling and typography errors, reviewing and implementing a range of solutions to tackle this problem through the use of deep learning and natural language processing. In particular methods such as Ngram Similarity, Rule-based estimation and Sequence-Sequence modelling will be explored to identify their given effectiveness within the detection and correction of misspelt text within the English language. The models are trained and validated through the use of synthetic data generation, with the use of real data from individuals with and without learning disabilities to benchmark the real-world effectiveness of the final model.

### Dependencies

The code was built in Python, within Windows 10 & Linux.

It was built within Python Version 3.9.18

A range of packages were used. Including the environment file to duplicate the environment. This can be found below within environment.ymal

The code was run within this environment. 

The key packages used can be found below.

| Package | Version |
| --- | --- |
| numpy | 1.26.4 |
| seaborn | 0.13.2 |
| SciPy | 1.13.0 |
| Scikit learn | 1.4.2 |
| Tensor Flow | 2.14.0 |
| Tensor Addons | 0.22.0 |
| matplotlib | 3.8.4 |
| nlkt | 3.8.1 |
| pyspellchecker | 0.8.1 |
| matplotlib | 3.6.3 |

### Installing

Everything is included within the code base. 
Install locally to a file. Next either manually build the environment or create a new environment, importing the provided yml file. The Databases are not included. 
Drag and drop the nzp files of the PnumoniaMNIST and PathMNIST datasets into the Dataset file.

### Executing program

The code is run through the command line. It can accept up to two arguments. If you wish to run the final model please run
```python
python3 main.py
```


## Hardware and training

Two machines were used to build, train and evaluate the models. The first is a XPS15 9500 and the second is a custom-built water-cooled tower. The second machine saw a significant improvement in training time. The specs of both machines, along with a comprehensive benchmark for the speed of the model are given below.

| Componenet | Cluster | Tower |
| --- | --- | --- |
| Processor | Intel(R) Xeon(R) Gold 6248 CPU | AMD Ryzen 7 5700x |
| OS | Linux |  Windows 10 |
| Ram | 512Gb | 32 GB |
| Graphics Card | NVidia 32Gb Tesla V100 | RTX 3070 |

## File Structure

Below is the file structure of the code base.

A contains the code for the data synthesis.

B contains all of the models built for this task. All of the graphics in the report can be found in Graphics, and all of the metric files (taken from the cluster), are stored within ClusterRuns. Models are saved within the training file, within B.

All of the datasets can be found within the Dataset Folder. As they are too large to upload to github, the code is setup to fetch a ZIP and unpack the ZIP files containing the final dataset locally.
```
├───A
│   └───Model
├───B
│   ├───Graphics
│   │   └───ClusterRuns
│   ├───StandAlone
│   └───training
└───Dataset
    ├───Mistakes
    └───Sentences
```

## Authors

Harry R J Softley-Graham  - SN: 19087176

## Aknowledgements


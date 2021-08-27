# On Quantitative Evaluations of Counterfactuals

## Install
To install required packages with conda, run the following command:

```bash 
> conda env create -f requirements.yml
```

## Code
The code contains all the evaluation metrics used in the paper as well as the models and the data.

To evaluate methods, you need to choose a config from the `configs` directory and to choose which metric to apply.
The code will then evaluate the chosen metrics on counterfactuals from all three methods (GB, GL, GEN) and store
the results in an appropriate subdirectory in `outputs`.
If you, e.g., want to run all metrics on the MNIST dataset, use the following command:

```bash
(cfeval) > python main.py --eval -c configs/mnist/mnist.ini -a
```

Afterwards you can enumerate the `directory` by
```bash
(cfeval) > python main.py --list
```
to get an output like the following:
```
> Listing dirs
000: ./output/celeba_makeup_[0]
001: ./output/fake_mnist_[0]
002: ./output/mnist_0_1_[0]
003: ./output/mnist_[0]
```

Now, results can be printed for the MNIST dataset (idx 3 above) by
```bash
(cfeval) > python main.py --print -c 3 
```
To get a result like
```
# # # # # # # # # # # # # # # # # # # # 
# MNIST
# # # # # # # # # # # # # # # # # # # # 
Method \ Metric    TargetClassValidity    ElasticNet    IM1          IM2             FID  Oracle
-----------------  ---------------------  ------------  -----------  -----------  ------  ------------
GB                 99.59 (0.13)           16.07 (0.18)  0.99 (0.00)  0.55 (0.01)   50.23  73.38 (0.87)
GL                 100.00 (0.00)          42.76 (0.31)  0.99 (0.00)  0.53 (0.00)  308.43  37.71 (0.95)
GEN                99.97 (0.03)           99.17 (0.58)  0.88 (0.00)  0.17 (0.00)   90.73  93.13 (0.50)
```

**Directory overview:** 

| File	| Description |  
| ----------------- | ----------------- |  
| `ckpts` 			| Contains all the (Keras) models used by the various metrics. |  
| `data` 			| Contains the data used, both counterfactual examples from GB, GL, and GEN, and original input data. |  
| `configs` 		| Contains config files specifying experimental details like dataset, normalization, etc. |  
| `data`			| Contains the data in numpy arrays. |  
| `dataset`			| Code for loading data. |  
| `evaluate`		| Implementations of all the metrics. |  
| `output`			| Directory to hold computed results. Directory already contains results from paper. |  
| `config.py`		| Reads config files from `configs` |  
| `constants.py`	| Method and metric names. |  
| `listing.py`		| Utility for indexing output dirs (see description below) |  
| `main.py`			| Main file to run all code through. |  
| `print_results.py`| Utillity function for printing results from json files in the `output` directory. |  
  


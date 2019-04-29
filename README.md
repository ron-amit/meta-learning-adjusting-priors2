# Implementation of the Meta-Learning-by-Adjusting-Priors algorithm in PyTorch 1.0

Implementation the paper R. Amit, R. Meir, “Meta-Learning by Adjusting Priors Based on Extended PAC-Bayes Theory”,  ICML 2018   [[paper](http://proceedings.mlr.press/v80/amit18a.html)]    [[Slides-Short](https://drive.google.com/file/d/1gDDrOi_f0Xs5t0NgFEQNbRDJ46BhgevT/view?usp=sharing)] [[Slides-long](https://drive.google.com/file/d/1ZmohF1FW2qneKRTit-AgjydhEYJKG-A1/view?usp=sharing)]   [[video](https://vimeo.com/294628795)]

## Prerequisites

- Python 3.5+
- [PyTorch 1.0+ with CUDA](http://pytorch.org)
- NumPy and Matplotlib


## Data
The data sets are downloaded automatically. Specify the main data path in the file 'Data_Path.py'

## Reproducing experiments in the paper:

* PriorMetaLearning/run_MPB_*.py   - Learns a prior from the obsereved (meta-training) tasks and use it to learn new (meta-test) tasks.
* Toy_Examples/Toy_Main.py -  Toy example of 2D  estimation.
* Single_Task/main_TwoTaskTransfer_PermuteLabels and  Single_Task/main_TwoTaskTransfer_PermutePixels.py -
run alternative tranfer methods.

* PriorMetaLearning/Analyze_Prior.py - Analysis of the weight uncertainty ine each layer of the learned prior (run after creating a prior with main_Meta_Bayes.py)

## Other experiments:

* Single_Task/main_single_standard.py         - Learn standard neural network in a single task.
* Single_Task/main_single_Bayes.py            - Learn stochastic neural network in a single task.

MAML code is based on: https://github.com/katerakelly/pytorch-maml

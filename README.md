# Robustness-Of-Fairness-2.0

Project developed by [Edward Small](https://www.admscentre.org.au/edward-small/) and [Dr. Wei Shao](https://swsamleo.github.io/wei_shao.github.io/) to measure the stability and consistency of models that pertain to give fair solutions.

<img src="https://user-images.githubusercontent.com/49641102/177453318-9f89f3b2-ac80-4921-a332-f83aba576f75.png" width="400"> <img src="https://user-images.githubusercontent.com/49641102/177453379-ab4c726b-ffc5-444b-bbe1-4b37647914eb.png" width="400">


If fairness of an AI $f$ over an input space $X$ is measured using a probability function $M(f, X)$, then robustness $R$ is measured via:

$R_k=\frac{M(f,X+\epsilon_k)}{M(f,X)}$ for $k>0$

where $\epsilon_k$ is a perturbation to the input space, and $k$ is the strength of this perturbation.

<img src="https://user-images.githubusercontent.com/49641102/177456336-d71d7cce-0110-44ca-8f20-dac351e4515b.png" width="800">

We desire combinations of models $f$, data $X$ and measures $M$ such that $R_k \leq 1$ for $k>0$.

To use the code, create a local copy using the git URL and install the required packages usint _pip install -r requirements.txt_. The code was written using python 3.9

The following is a list of command line arguments:

* *--dataset 'dataset name'* is the name of the dataset you wish to run (from the current list of accessible data sets). See [the folktables](https://github.com/zykls/folktables) for details of the ACS data sets. List of pre-built data sets is in data_util/fetch_data.py in the get_data function.
* *--train_constraint 'constraint'* is the fairness training constraints. dp is demographic parity, eo is equalised odds, fp is false positive, and tp is true positive.
* *--output_dir* 'name'* is the name of the output directory for all the data and plots
* *--min_noise* 'number'* is the minimum number for $k$.
* *--max_noise* 'number'* is the maximum number for $k$.
* *--noise_iters 'number'* is the number of iterations per value of $k$ (this is because some methods involve an element of randomness).
* *--model_iters 'number'* is the maximum number of iterations to fit each model.
* *--model_type 'model name'* is the type of model we wish to use (eg SVC is supper vectors, LR is logistic regression, etc).
* *--step_size 'number* is the step size between each noise level.
* *--roc 'True/False'* decides whether to create and display the ROC curve.

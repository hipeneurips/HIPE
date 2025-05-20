# Informed Initialization for Bayesian Optimization and Active Learning - Official Repository
This repository implements and compares several initialization methods for Bayesian Optimization (BO), including the proposed HIPE, NIPV, BALD, LHS-Beta, Random and Sobol.

---

## Installation

Create an environment and install dependencies:

```pip install -r requirements.txt```


## Running experiments
```python main.py```

will run Sobol (default) on Hartmann (6D, also default).


```python main.py init=hipe objective=hartmann4_8 experiment.budget=48 acq_opt.q=16 seed=4242```

To run active learning instead of BO, enable it with ```task=al```:
 
```python main.py objective=ishigami init=hipe experiment.budget=48 acq_opt.q=16 task=al```


Additional example with LCBench and fixed (zero) noise in the model:

```python main.py objective=lcbench_car init=hipe model.fixed_noise=1 experiment.budget=48 acq_opt.q=16```

will run HIPE on embeddedd Hartmann (4D in 8D), with a budget of 48 and batch_size of 16.



## Plots

To start plotting, first ```export PYTHONPATH=$PWD```.

By default, all experiments appear in 

```results/test/function_name/method_name/seed/runfiles.json```

 (one named ```init.json``` and one named either ```bo.json``` or ```al.json```). When running plot scripts, all functions and methods within the folder are plotted by default, so 
 
 ```python scripts/regret.py results/path_to_results```
 
  will plot the inference regret for all functions and methods inside ```results/path_to_results```.

```python scripts/regret.py results/path_to_results --functions=ackley4,hartmann4_8 --methods=sobol,hipe```

will plot the inference regret, per batch, for the aforementioned functions and methods. ```relative_ranking=1``` will include the relative ranking plots.

```python scripts/active.py results/path_to_results --functions=ackley4,hartmann4_8 --methods=sobol,hipe --metric=MLL```

will plot the negative log likelihood on test points per batch, and ```--metric=RMSE``` does the same for RMSE (if an AL experiment has been run). ```relative_ranking=1``` once again enables relative rankings. 


## All options

##### Benchmarks
The following benchmarks and methods are available to run:

Synthetic: ```ackley4,hartmann4,hartmann4_8,hartmann6,hartmann6_12, ishigami```

LCBench: ```lcbench_australian_al, lcbench_car_al,lcbench_car,lcbench_higgs,lcbench_mini,lcbench_segment,lcbench_fashionmnist```

SVM: ```svm_20,svm_40```

For SVM, make sure to append ```eval=0``` to the end of the command line arguments to ensure that MLL and RMSE are not computed (250 extra function evaluations per batch).

##### Initialization Methods

```hipe,nipv,bald,random,sobol,lhsbeta```

See each benchmark & method's ```yaml``` config file for additinal option.



For additional inquiries, please start an issue in the repository and we will make sure to respond urgently.
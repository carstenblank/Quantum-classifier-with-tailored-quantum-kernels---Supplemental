# Quantum classifier with tailored quantum kernel

## Supplementary Material

### How to use the code

In order to have the computation be run in the background without worrying too much about the timing
that is necessary to send experiment and simulation after each other and to use as much parallelism as 
possible we are using Dask.

To start a single worker cluster just call
```bash 
bash ./start_single_dask_cluster.sh
```
This will install all necessary python requirements before and then start a dask scheduler and one 
single dask worker with a disabled "nanny", which means the processes that are running in the worker
are allowed to spawn processes themselves. This is usually forbidden, but since qiskit uses the process
library this is a must.

After that you are able to run the code. In order to get a feeling use one of our notebooks.

### Setup Notebooks

The notebooks need the code imported. Therefore it makes sense that you add to the python path the directory `dc_qiskit_swaptest_classifier` before using the code or starting e.g. jupyter-lab
```bash
PYTHONPATH="${PWD}/dc_qiskit_swaptest_classifier/:${PWD}/experiment_results/" jupyter-lab
```
this will load the absolute path of the code into the python path so that imports work.

### Setup IBMQ

```python
import qiskit
QX_TOKEN = "..."
QX_URL = "https://quantumexperience.ng.bluemix.net/api"
qiskit.IBMQ.enable_account(QX_TOKEN, QX_URL)

```
where you need to set the `QX_TOKEN` accordingly to the one you can get from IBM-Q.
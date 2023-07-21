

<div align="center">

# Object_Detection
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://github.com/pyscaffold/pyscaffoldext-dsproject"><img alt="Template" src="https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

# Description
This is a custom pipeline dedicated to solving object detection problems and is derived from Quicksetup-ai, a flexible template as a quick setup for deep learning projects in research. The pipeline comes with preloaded Faster RCNN architecture and standard computer vision modules. 

# Quickstart

## Create the pipeline environment and install the object_detection package
Before using the template, one needs to install the project as a package.
* First, create a virtual environment.
> You can either do it with conda (preferred) or venv.
* Then, activate the environment
* Finally, install the project as a package. Run:
```
pip install -e .
```
## Run the organoid detection example
This pipeline comes with a toy example to perform object detection on sample of intestinal organoid images. 

To run the training pipeline, simply run:
```
python scripts/train.py

```
To run the testing pipeline, simply run:
```
python scripts/test.py
```
Or, if you want to submit the training job to a submit (resp. interactive) cluster node via slurm, run:
```
sbatch job_submission.sbatch
# or sbatch job_submission_interactive.sbatch
```
> * The experiments, evaluations, etc., are stored under the `logs` directory.
> * The default experiments tracking system is mlflow. The `mlruns` directory is contained in `logs`. To view a user friendly view of the experiments, run:
> ```
> # make sure you are inside logs (where mlruns is located)
> mlflow ui --host 0000
> ```
> * When evaluating (running `test.py`), make sure you give the correct checkpoint path in `configs/test.yaml`

The predictions are saved in this path as a json file `object_detection/scripts/data/predictions`

### Citation
The sample dataset is acquired from the below research,
`Kassis, Timothy, et al. "OrgaQuant: human intestinal organoid localization and quantification using deep convolutional neural networks." Scientific reports 9.1 (2019): 12479.`
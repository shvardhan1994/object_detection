

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

### 1. Install dependencies

#### 1.1 Installing dependencies on your local machine (with CPU or GPU) and creating conda environment.

```bash
# clone project
git clone https://github.com/HelmholtzAI-Consultants-Munich/object_detection.git
cd object_detection

# [OPTIONAL] create conda environment
conda create -n venv python>=3.8
conda activate venv

# Ensure that the lines between 51 and 93 are not commented in the setup.cfg file.

# install requirements and "object_detection module"
pip install -e .
```

#### 1.2 Installing dependencies on JUWELS and creating a virtual environment.

```bash
# clone project
git clone https://github.com/HelmholtzAI-Consultants-Munich/object_detection.git
cd object_detection

# [OPTIONAL] create virtual environment
git clone https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template.git

# Edit `config.sh` to change name and location of the venv if required.
# Edit `modules.sh` to change the modules loaded prior to the creation of the venv.
# Copy the lines between 51 and 93 of `setup.cfg` to `sc_venv_template/requirements.txt` and comment the same lines between 51 and 93 in the setup.cfg file.

# Create the environment with 
bash setup.sh

# Create a Kernel to use Jupyter notebooks on JUWELS
bash create_kernel.sh

# Activate the virtual environment with 
source sc_venv_template/activate.sh

# Once the environment is created, execute 
pip install -e .

```
## Run the organoid detection example

**Note**: This pipeline performs object detection on intestinal organoid images. The dataset is split into train, validation and test set. The dataset has to be saved in object_detection/data/orgaquant folder. The data folder should follow below structure.

├── object_detection                             
│   ├── data                               
│       ├── orgaquant                             
│           ├── train
            ├── val
            ├── test
            ├── train_labels.csv
            ├── val_labels.csv
            ├── test_labels.csv
        ├── predictions
        ├── metrics

The .csv files contains the annotations for the bounding boxes of organoids present in the input images. The .csv file has below structure,

```x1 | y1 | x2 | y2 | class_name | path```


To run the training pipeline, simply run:
```
python scripts/train.py

```

Or, if you want to submit the training job to a submit (resp. interactive) cluster node via slurm, run:
```
sbatch job_submission.sbatch
# or sbatch job_submission_interactive.sbatch
```
By default, the training pipeline also runs the testing pipeline on the testset based on the best model checkpoint during training.

If you want to run just testing pipeline exclusively, make sure you give the correct checkpoint path in `configs/test.yaml` and then simply run:
```
python scripts/test.py

```
The predictions are saved by default in this path as a json file `object_detection/scripts/data/orgaquant/predictions`

To compute the evaluation metrics on the testset (this pipeline calculates Precisin-Recall values for different box score thresholds and saves them as an array in `object_detection/data/orgaquant/metrics`), specify the path to the predictions in .json format as mentioned previously in the inference script in `object_detection/notebooks/inference.py`, then simply run:

```
python inference.py

```


> * The experiments, evaluations, etc., are stored under the `logs` directory.
> * The default experiments tracking system is mlflow. The `mlruns` directory is contained in `logs`. To view a user friendly view of the experiments, run:
> ```
> # make sure you are inside logs (where mlruns is located)
> mlflow ui --host 0000
> ```
> * When evaluating (running `test.py`), make sure you give the correct checkpoint path in `configs/test.yaml`



### References
The sample dataset is acquired from the below research,
`Kassis, Timothy, et al. "OrgaQuant: human intestinal organoid localization and quantification using deep convolutional neural networks." Scientific reports 9.1 (2019): 12479.`
# Kidney-ct-classification
Kidney Ct Scan Image Classification using Pytorch

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

## Dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/andcoul/kidney-ct-classification-pytorch.mlflow \
MLFLOW_TRACKING_USERNAME=andcoul \
MLFLOW_TRACKING_PASSWORD=3c2d382f23858d4d2a20e9d5f187d6bdef1e83c2 \
python script.py

Export the following into the environment variables

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/andcoul/kidney-ct-classification-pytorch.mlflow
export MLFLOW_TRACKING_USERNAME=andcoul
export MLFLOW_TRACKING_PASSWORD=3c2d382f23858d4d2a20e9d5f187d6bdef1e83c2

```

### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


### About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



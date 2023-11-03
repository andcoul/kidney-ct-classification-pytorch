# Kidney-Ct-Scan-classification
Kidney Ct Scan Image Classification using Pytorch

## Workflows:

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

## AWS-CICD-Deployment-with-Github-Actions

### 1. Login to AWS console.

### 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
### 3. Create ECR repo to store/save docker image
    - Save the URI: 737647635436.dkr.ecr.us-east-1.amazonaws.com/kidney

	
### 4. Create EC2 machine (Ubuntu) 

### 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
### 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


### 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app



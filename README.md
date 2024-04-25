# Deploy_Model
Deploy Hugging Face Model to GCP 
Referenced: https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai

## Create a virtual environment
In the root directory of the project, create a virtual environment using the following command:
```python3 -m venv venv```
```source venv/bin/activate```

## Find the GPU using homework 2
The Find_GPU.ipynb file is used to find the GPU. Follow the command in the notebook to find available GPU.
At the end of the notebook, a list of available GPU is displayed.
We will use this to be the accelerator of the model.

## Usage of external resources
- We will use the Docker image from alvarobartt as base image and then build the predictor to push to GCP.
- We will use the Hugging Face Predictor function in our docker image so that we are able to load the pipeline.

## The Aggregatino of Command to Deploy the Model
There are two ipynb jupyter notebook files that are used to deploy the model to Vertex ai, simply follow the command and you will be able to
deploy the model to Vertex AI.
- Deploy_model.ipynb -> this deploys the model to Vertex with no accelerator
- Deploy_model_gpu.ipynb -> this deploys the model to Vertex with accelerator after finding gpu with Find_GPU.ipynb

## Steps to run deploy the model
1. go to the deploy_model.ipynb and run the commands in the notebook
2. Follow the instruction one by one and you will be able to deploy the model to Vertex AI
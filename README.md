## **Dreambooth API**
This repository contains the API to train Stable Diffusion models on Sagemaker using Dreambooth.
The API is built using FastAPI framework, and it is handled by Mangum module in order to deploy it in AWS Lambda + AWS API Gateway.
This training requires a GPU with at least 16BG of VRAM (e.g. "ml.g5.xlarge" instance)

### **Repository Structure**
```
app
│   main.py
│   
└───src
    │
    └───training
    │   │    train.py
    │   │    sagemaker_train.py
    │   
    └───model
    │   │    models.py
    │
    └───inference
        │   inference.py
        │   sagemaker_inference.py
```

**main.py:** API code. Used FastAPI framework.

**train.py:** Sagemaker training entrypoint. This script is run during each training job. It loads the training data, runs the training and save the final model.

**sagemaker_train.py:** Contains the SagaMakerTraining class that gather all the required methods to perform the training.

**models.py:** Contains the PyDantic models needed to validate the data on FastAPI.

**inference.py:** Sagemaker inference entrypoint.

### **Deployment**
The API is currently deployed in the following url: https://3f041ecl7b.execute-api.us-east-1.amazonaws.com/dev

It is deployed using AWS Lambda + AWS API Gateway, which allows serverless scalability at a low cost.




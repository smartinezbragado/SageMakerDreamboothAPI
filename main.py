import os
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from src.training.sagemaker_train import SageMakerTrain
from src.model.models import TrainingConfig, DeleteInput
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


load_dotenv()
s3_client = boto3.client('s3')

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.post('/token')
async def authentication_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return {'access_token': form_data.username + 'token'}

@app.get('/')
async def index(token: str = Depends(oauth2_scheme)):
    return {'token': token}


@app.post('/models/training')
async def train_sagemaker(train_config: TrainingConfig) -> dict:
    sm = SageMakerTrain(train_config.dict())
    sm.run()
    return {'job_name': sm.job_name}


@app.get('/models/status')
async def get_training_status(user_id: str, model_id: str) -> dict:
    sm = SageMakerTrain({
        "user_id": user_id,
        "model_id": model_id
    })
    training_status = sm.sess.describe_training_job(sm.job_name)['TrainingJobStatus']
    return {'training_status': training_status}


@app.get('/models/list')
async def list_models(user_id: str) -> dict:
    result = s3_client.list_objects(
        Bucket=os.getenv('SAGEMAKER_BUCKET'), Prefix=user_id, Delimiter='/'
    )
    models = [i['Prefix'].split('/')[0] for i in result['CommonPrefixes']]
    return {user_id: {
            'models': models
            }
        }

@app.post('/models/delete/')
async def delete_models(payload: DeleteInput) -> None:
    model_name = f"{payload.get('user_id')}-{payload.get('model_id')}"
    response = s3_client.list_objects_v2(
        Bucket=os.getenv('SAGEMAKER_BUCKET'), Prefix=model_name
    )

    for object in response['Contents']:
        print('Deleting', object['Key'])
        s3_client.delete_object(
            Bucket=os.getenv('SAGEMAKER_BUCKET'), Key=object['Key']
        )

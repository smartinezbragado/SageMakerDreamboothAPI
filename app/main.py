import os
import boto3
import logging
from mangum import Mangum
from dotenv import load_dotenv
from fastapi import FastAPI
from botocore.exceptions import ClientError
from app.src.utils.utils import *
from app.src.training.sagemaker_train import SageMakerTrain
from app.src.model.models import TrainingConfig, DeleteInput, InferenceConfig

load_dotenv()
logger = logging.getLogger(__name__)
s3_client = boto3.client('s3', region_name="us-east-1")

app = FastAPI()


@app.get('/')
async def index():
    return {'Message': "Welcome to dreambooth api"}


@app.post('/models/training')
async def send_training_job_to_sagemaker(payload: TrainingConfig) -> dict:
    train_cfg = payload.dict()
    train_cfg['hyperparameters']['instance_prompt'] = train_cfg.get('hyperparameters').get('instance_prompt').replace(' ', '_')
    train_cfg['hyperparameters']['class_prompt'] = train_cfg.get('hyperparameters').get('class_prompt').replace(' ', '_')
    train_cfg['hyperparameters']['prompt'] = train_cfg.get('hyperparameters').get('prompt').replace(' ', '_')
    sm = SageMakerTrain(train_cfg)

    try:
        sm.train()
        return {'job_name': sm.job_name,'status': 'successful'}

    except ClientError as e:
        logger.info(e)
        error_code = e.response.get('Error', {}).get('Code')

        if error_code == 'ResourceInUse':
            return {'error': f'Already exist a job name {sm.job_name}'}

        elif error_code == 'ResourceLimitExceeded':
            sm.send_job_to_queue()
            return {'job_name': sm.job_name, 'status': 'queue', 'jobs_in_queue': sm._get_jobs_in_queue()}


@app.get('/models/status')
async def get_sagemaker_training_job_status(user_id: str, model_id: str) -> dict:
    sm = SageMakerTrain({"user_id": user_id,"model_id": model_id})
    try:
        training_status = sm.sess.describe_training_job(sm.job_name)['SecondaryStatusTransitions']

        if training_status[-1]['Status'] != 'Completed':
            return {
                'job_name': sm.job_name, 
                'status': training_status[-1]['Status']
            }
        
        elif training_status[-1]['Status'] == 'Completed':
            extract_images_from_s3_tar_file(sm.job_name)
            url = get_url_from_s3_path(f"s3://{os.getenv('SAGEMAKER_BUCKET')}/{sm.job_name}/output/images/")
            
            return {
                'job_name': sm.job_name, 
                'status': training_status[-1]['Status'],
                'images': [
                    os.path.join(url, f'{n}.jpeg') for n in range(len(os.listdir('/tmp/images')))
                ]
            }
        
    except ClientError:
        return {
            'job_name': sm.job_name, 
            'status': 'queue', 
            'jobs_in_queue': sm._get_jobs_in_queue()
        }


@app.get('/models/list')
async def list_user_models(user_id: str) -> dict:
    result = s3_client.list_objects(
        Bucket=os.getenv('SAGEMAKER_BUCKET'), Prefix=user_id, Delimiter='/'
    )
    models = [i['Prefix'].split('/')[0] for i in result['CommonPrefixes']]
    return {user_id: {'models': models}}


@app.post('/models/delete/')
async def delete_user_model(payload: DeleteInput) -> None:
    delete_config = payload.dict()
    model_name = f"{delete_config.get('user_id')}-{delete_config.get('model_id')}"
    response = s3_client.list_objects_v2(
        Bucket=os.getenv('SAGEMAKER_BUCKET'), Prefix=model_name
    )
    try:
        for object in response['Contents']:
            print('Deleting', object['Key'])
            s3_client.delete_object(Bucket=os.getenv('SAGEMAKER_BUCKET'), Key=object['Key'])

        return {'model_name': model_name, 'status':'successful'}

    except Exception as e:
        logger.info(e)
        return {'model_name': model_name, 'status':'error'}
    

@app.post('/models/inference')
async def make_inference(payload: InferenceConfig):
    return {
        'images': [
        's3://dreambooth-testing03621-dev/07a4ef2cd994440f8f478d14c9329acc/image_cropper_0A903414-DAFC-42AC-8703-B9FCF297BF04-12140-000007BDFF7C0090.jpg',
        's3://dreambooth-testing03621-dev/07a4ef2cd994440f8f478d14c9329acc/image_cropper_10E455CC-3023-48D6-B94A-F0109F130DAE-12140-000007A8D1168AA1.jpg'
        ]
    }

handler = Mangum(app)

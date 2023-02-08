import os
import boto3
import logging
import sagemaker
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace

load_dotenv()

logger = logging.getLogger(__name__)

class SageMakerTrain:

    def __init__(self, cfg: dict):

        self.cfg = cfg

        # User & model ids
        self.user_id = self.cfg.get('user_id')
        self.model_id = self.cfg.get('model_id')
        self.job_name = f"{self.user_id}-{self.model_id}"

        # Sagemaker entrypoint & config
        self.source_dir = self.cfg.get('source_dir')
        self.entry_point = self.cfg.get('entry_point')
        self.hyperparameters = self.cfg.get('hyperparameters')

        # Sagemaker instances config
        self.instance_type = self.cfg.get('instance_type')
        self.instance_count = self.cfg.get('instance_count')

        logger.info(f'Job name: {self.job_name}')
        logger.info(f'Instance type: {self.instance_type}')
        logger.info(f'Instance count: {self.instance_count}')

        self.__init_session()

    def __init_session(self):
        """Initiates SageMaker session"""
        sm_boto = boto3.client("sagemaker")

        sess = sagemaker.Session(sagemaker_client=sm_boto)
        sagemaker_session_bucket = None

        if sagemaker_session_bucket is None and sess is not None:
            # set to default bucket if a bucket name is not given
            self.sagemaker_session_bucket = sess.default_bucket()

        self.sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    def _set_estimator(self):
        """Defines the HuggingFace estimator configuration"""
        self.huggingface_estimator = HuggingFace(
            entry_point=self.entry_point,
            source_dir=self.source_dir,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=os.getenv('SAGEMAKER_ROLE'),
            transformers_version="4.17",
            pytorch_version="1.10",
            py_version="py38",
            max_run=36000,
            sagemaker_session=self.sess,
            hyperparameters=self.hyperparameters
        )

    def _train(self) -> None:
        """Applies Dreambooth over the folder images"""
        logger.info(f'Starting training job {self.job_name}')
        self.huggingface_estimator.fit(
            inputs={
                "train": f"s3://{os.getenv('DEVELOPMENT_BUCKET')}/{self.user_id}/",
                'test': f"s3://{os.getenv('DEVELOPMENT_BUCKET')}/{self.user_id}/"
            },
            wait=False, 
            job_name=self.job_name
        )

    def run(self) -> None:
        self._set_estimator()
        self._train()
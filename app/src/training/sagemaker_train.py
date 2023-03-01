import os
import boto3
import logging
import sagemaker
import stepfunctions
from dotenv import load_dotenv
from sagemaker.huggingface import HuggingFace
from stepfunctions.workflow import Workflow
from stepfunctions.steps.states import Retry
from stepfunctions.steps import TrainingStep, Chain

load_dotenv()
logger = logging.getLogger(__name__)
stepfunctions.set_stream_logger(level=logging.INFO)

stepfunctions_client = boto3.client('stepfunctions')

class SageMakerTrain:

    def __init__(self, cfg: dict):

        self.cfg = cfg

        # User & model ids
        self.user_id = self.cfg.get('user_id')
        self.model_id = self.cfg.get('model_id')
        self.job_name = f"{self.user_id}-{self.model_id}"
        self.training_path = f"s3://{os.getenv('DEVELOPMENT_BUCKET')}/{self.user_id}/"

        # Sagemaker entrypoint & config
        self.source_dir = self.cfg.get('source_dir')
        self.entry_point = self.cfg.get('entry_point')
        self.hyperparameters = self.cfg.get('hyperparameters')

        # Sagemaker instances config
        self.instance_type = self.cfg.get('instance_type')
        self.instance_count = self.cfg.get('instance_count')

        # Step functions config
        self.n_retries = self.cfg.get('n_retries', 120)
        self.retry_interval = self.cfg.get('retry_interval', 60)

        logger.info(f'Job name: {self.job_name}')
        logger.info(f'Instance type: {self.instance_type}')
        logger.info(f'Instance count: {self.instance_count}')
        logger.info(f'Hyperparamers: {self.hyperparameters}')

        self.__init_session()

    def __init_session(self):
        """Initiates SageMaker session"""
        sm_boto = boto3.client("sagemaker", region_name="us-east-1")

        sess = sagemaker.Session(sagemaker_client=sm_boto)
        sagemaker_session_bucket = None

        if sagemaker_session_bucket is None and sess is not None:
            # set to default bucket if a bucket name is not given
            self.sagemaker_session_bucket = sess.default_bucket()

        self.sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    def _get_jobs_in_queue(self):
        """Returns the number of active jobs in the queue"""
        state_machines = stepfunctions_client.list_state_machines().get('stateMachines')

        if len(state_machines) == 0:
            return 0

        else:
            n_jobs = 0
            for machine in state_machines:
                arn = machine.get('stateMachineArn')
                executions = stepfunctions_client.list_executions(stateMachineArn=arn).get('executions')
                status = executions[-1].get('status')
                if status == "RUNNING":
                    n_jobs += 1

            return n_jobs

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

    def send_job_to_queue(self):
        """Send the job to a queue managed by AWS Step Functions"""
        logger.info('Sending job to the queue')
        retry = Retry(
            error_equals = ["States.ALL"],
            interval_seconds = self.retry_interval,
            max_attempts = self.n_retries
        )
        training_step = TrainingStep(
            state_id='training',
            job_name=self.job_name,
            estimator=self.huggingface_estimator,
            data={
                "train": self.training_path,
                "test": self.training_path
            },
            wait_for_completion=False
        )
        training_step.add_retry(retry)
        workflow_definition = Chain([training_step])
        logger.info(f"Steps function role: {os.getenv('EXECUTION_WORKFLOW_ROLE')}")
        workflow = Workflow(
            name=self.job_name,
            definition=workflow_definition,
            role= os.getenv('EXECUTION_WORKFLOW_ROLE'),
        )
        workflow.create()
        workflow.execute()

    def train(self) -> None:
        """Applies Dreambooth over the folder images"""
        logger.info(f'Starting training job {self.job_name}')
        self._set_estimator()
        self.huggingface_estimator.fit(
            inputs={"train": self.training_path, 'test': self.training_path},
            wait=False, job_name=self.job_name
        )

import os
from typing import Optional
from pydantic import BaseModel


class Hyperparameters(BaseModel):
    instance_prompt: Optional[str] = None 
    resolution: int = 768
    learning_rate: float = 5e-6
    lr_warmup_steps: int = 0
    num_class_images: int = 200
    max_train_steps: int = 800


class TrainingConfig(BaseModel):
    user_id: str
    model_id: str
    entry_point: str = "train.py"
    source_dir: str = os.path.join('..', 'training')
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    hyperparameters: Hyperparameters


class InferenceInput(BaseModel):
    prompt: str
    number: int
    guidance_scale: float = 7.5
    num_inference_steps: int = 50


class DeleteInput(BaseModel):
    user_id: str
    model_id: str
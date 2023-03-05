
import os
from typing import Optional
from pydantic import BaseModel


class Hyperparameters(BaseModel):
    instance_prompt: str
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2"
    resolution: int = 768
    train_batch_size: int = 1
    learning_rate: float = 5e-6
    lr_warmup_steps: int = 0
    num_class_images: int = 200
    max_train_steps: int = 800
    center_crop: bool = False
    ckpt: bool = False
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1
    class_data_dir: str = os.path.join('opt' ,'ml', 'model', 'class_images')
    class_prompt: str = "A photo of a person"
    train_text_encoder: bool = False
    prompt: Optional[str] = None
    number: int = 2
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    widht: int = 768
    height: int = 768
    hub_token: Optional[str] = os.getenv('HUB_TOKEN')
    hub_model_id: Optional[str] = None
    push_to_hub: bool = False


class TrainingConfig(BaseModel):
    user_id: str
    model_id: str
    entry_point: str = "train.py"
    source_dir: str = os.path.join('app', 'src', 'training')
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    n_retries: int = 120
    retry_interval: int = 60
    hyperparameters: Hyperparameters


class InferenceConfig(BaseModel):
    user_id: str
    model_id: str 
    prompt: str
    number: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    height: int = 768
    width: int = 768
    negative_prompt: Optional[str] = None


class DeleteInput(BaseModel):
    user_id: str
    model_id: str
"""
Training script for Hugging Face SageMaker Estimator
"""
import argparse
import logging
import os
import sys

os.system('pip install bitsandbytes')
os.system('pip install setuptools==59.5.0')


import glob    


def convert_str_to_bool(arg: str):
    if arg.lower() == 'false':
        return False
    elif arg.lower() == 'true':
        return True

# TODO: script comes from example, refactor it according to needs
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    model_path = os.path.join('opt' ,'ml', 'model')

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--instance_prompt", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--num_class_images", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--ckpt", default=False)
    parser.add_argument("--with_prior_preservation", default=False)
    parser.add_argument("--prior_loss_weight", type=float, default=1)
    parser.add_argument("--class_data_dir", default=os.path.join(model_path, 'class_images'))
    parser.add_argument("--class_prompt", default="A photo of a person")
    parser.add_argument("--train_text_encoder", default=False)
    parser.add_argument("--prompt", default="A photo of a man in the moon")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--number", type=int, default=2)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--center_crop", default=False)
    parser.add_argument("--checkpointing_steps", type=int, default=1500)
    parser.add_argument("--push_to_hub", default=True)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--hub_model_id", type=str)


    print(os.environ["SM_NUM_GPUS"])

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Convert string to bool
    args.ckpt = convert_str_to_bool(args.ckpt)
    args.with_prior_preservation = convert_str_to_bool(args.with_prior_preservation)
    args.train_text_encoder = convert_str_to_bool(args.train_text_encoder)
    args.push_to_hub = convert_str_to_bool(args.push_to_hub)

    # Preprocess string inputs
    args.instance_prompt = args.instance_prompt.replace('_', ' ')
    args.class_prompt = args.class_prompt.replace('_', ' ')
    args.prompt = args.prompt.replace('_', ' ')

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    #??Donwload diffusers repo & install dependencies
    os.system('git clone https://github.com/huggingface/diffusers')
    os.system('pip install -e ./diffusers/')
    os.system('pip install -r ./diffusers/examples/dreambooth/requirements.txt')

    # Remove useless files to free memory space
    os.system('rm -r ./diffusers/docs/')
    os.system('rm -r ./diffusers/tests/')
    os.system('rm -r ./diffusers/examples/rl/')
    os.system('rm -r ./diffusers/examples/community/')
    os.system('rm -r ./diffusers/examples/research_projects/')
    os.system('rm -r ./diffusers/examples/inference/')
    os.system('rm -r ./diffusers/examples/text_to_image/')
    os.system('rm -r ./diffusers/examples/textual_inversion/')
    os.system('rm -r ./diffusers/examples/unconditional_image_generation/')
    os.system('rm -r ./diffusers/docker/')
    os.system('rm -r ./finetuned-model/')
    os.system('rm ./.ipynb_checkpoints')
    os.system('rm -r ./diffusers/.git/')
    os.system('rm -r ./diffusers/.github/')

    for i in glob.iglob('./diffusers/**/__pycache__', recursive=True):
        os.system(f'rm -r {i}')
    
    logger.info(args.training_dir)
    logger.info(args.model_dir)

    if args.push_to_hub:
        os.system("pip install --upgrade huggingface_hub")
        from huggingface_hub import create_repo, Repository

        model_path = os.path.join('tmp')   
        repo_name = f"Vinwapgames/{args.hub_model_id}"
        create_repo(repo_id=repo_name, private=True, token=args.hub_token)
        repo = Repository(model_path, clone_from=repo_name, token=args.hub_token) 

    else:
        model_path = args.model_dir

    os.system('pip install protobuf==3.20.* --upgrade')
    logger.info(args)

    if args.train_text_encoder:
        logger.info('Started dreambooth with train_text_encoder and prior preservation')

        os.system(f"""
        accelerate launch --gpu_ids=0 --num_processes=1 --num_machines=1 --dynamo_backend='no' --mixed_precision='fp16' \
            diffusers/examples/dreambooth/train_dreambooth.py \
            --pretrained_model_name_or_path={args.pretrained_model_name_or_path}  \
            --train_text_encoder \
            --instance_data_dir={args.training_dir} \
            --class_data_dir={args.class_data_dir} \
            --output_dir={model_path} \
            --with_prior_preservation --prior_loss_weight={args.prior_loss_weight} \
            --instance_prompt="{args.instance_prompt}" \
            --class_prompt="{args.class_prompt}" \
            --resolution={args.resolution} \
            --train_batch_size={args.train_batch_size} \
            --gradient_accumulation_steps={args.gradient_accumulation_steps} --gradient_checkpointing \
            --use_8bit_adam \
            --learning_rate={args.learning_rate} \
            --lr_scheduler={args.lr_scheduler} \
            --lr_warmup_steps={args.lr_warmup_steps} \
            --num_class_images={args.num_class_images} \
            --max_train_steps={args.max_train_steps} \
            --checkpointing_steps={args.checkpointing_steps} \
            --center_crop \
        """)

    if args.with_prior_preservation:
        logger.info('Started dreambooth with prior preservation')

        os.system(f"""
        accelerate launch --gpu_ids=0 --num_processes=1 --num_machines=1 --dynamo_backend='no' --mixed_precision='fp16' \
            diffusers/examples/dreambooth/train_dreambooth.py \
            --pretrained_model_name_or_path={args.pretrained_model_name_or_path}  \
            --instance_data_dir={args.training_dir} \
            --class_data_dir={args.class_data_dir} \
            --output_dir={model_path} \
            --with_prior_preservation --prior_loss_weight={args.prior_loss_weight} \
            --instance_prompt="{args.instance_prompt}" \
            --class_prompt="{args.class_prompt}" \
            --resolution={args.resolution} \
            --train_batch_size={args.train_batch_size} \
            --gradient_accumulation_steps={args.gradient_accumulation_steps} --gradient_checkpointing \
            --use_8bit_adam \
            --learning_rate={args.learning_rate} \
            --lr_scheduler={args.lr_scheduler} \
            --lr_warmup_steps={args.lr_warmup_steps} \
            --num_class_images={args.num_class_images} \
            --max_train_steps={args.max_train_steps} \
            --checkpointing_steps={args.checkpointing_steps} \
            --center_crop \
        """)

    else:
        logger.info('Started dreambooth')

        os.system(f"""
        accelerate launch --gpu_ids=0 --num_processes=1 --num_machines=1 --dynamo_backend='no' --mixed_precision='fp16' \
            diffusers/examples/dreambooth/train_dreambooth.py \
            --pretrained_model_name_or_path={args.pretrained_model_name_or_path}  \
            --instance_data_dir={args.training_dir} \
            --output_dir={model_path} \
            --instance_prompt="{args.instance_prompt}" \
            --resolution={args.resolution} \
            --train_batch_size={args.train_batch_size} \
            --gradient_accumulation_steps={args.gradient_accumulation_steps} --gradient_checkpointing \
            --use_8bit_adam \
            --learning_rate={args.learning_rate} \
            --lr_scheduler={args.lr_scheduler} \
            --lr_warmup_steps={args.lr_warmup_steps} \
            --max_train_steps={args.max_train_steps} \
            --center_crop \
            --checkpointing_steps={args.checkpointing_steps}
        """)
    
    os.system('pip install diffusers==0.13.0')
    import torch
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    # model.enable_attention_slicing()
    model = model.to("cuda")

    inference_input = {
        "prompt": [args.prompt] * args.number,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width
    }
    logger.info(inference_input)

    images = model(**inference_input)["images"]
    os.mkdir(os.path.join(args.model_dir, 'images'))

    for n, i in enumerate(images):
        i.save(os.path.join(args.model_dir, 'images', f'{n}.jpeg'))

    if args.ckpt:    
        os.system("pip install safetensors")

        os.system(f"""
        python diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py \
            --model_path={model_path} \
            --checkpoint_path={os.path.join(model_path, 'model.ckpt')} \
            --half
        """)

        os.system(f"rm -r {os.path.join(model_path, 'vae')}")
        os.system(f"rm -r {os.path.join(model_path, 'unet')}")
        os.system(f"rm -r {os.path.join(model_path, 'scheduler')}")
        os.system(f"rm -r {os.path.join(model_path, 'feature_extractor')}")
        os.system(f"rm -r {os.path.join(model_path, 'tokenizer')}")
        os.system(f"rm -r {os.path.join(model_path, 'text_encoder')}")

    if args.push_to_hub:
        logger.info('Pushing to hub')
        repo.push_to_hub(commit_message="End of training", blocking=True)

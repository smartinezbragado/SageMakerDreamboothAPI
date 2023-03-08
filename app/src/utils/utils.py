import os
import time
import boto3
import tarfile

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')


def extract_files_from_targz(path: str) -> None:
    tar = tarfile.open(path, "r:gz")
    tar.extractall('/tmp/')
    tar.close()


def upload_folder_to_s3(path: str, job_name: str) -> None:
    for root, dirs, files in os.walk(path):
        for file in files:
            s3_client.upload_file(
                os.path.join(root,file), 
                os.getenv('SAGEMAKER_BUCKET'), 
                os.path.join('inference', job_name, 'images', file)
            )


def extract_images_from_s3_tar_file(job_name: str):
    try:
        os.mkdir('tmp')
    except Exception as e:
        print(e)

    s3_client.download_file(
        os.getenv('SAGEMAKER_BUCKET'),
        f'{job_name}/output/model.tar.gz', 
        '/tmp/model.tar.gz'
    )
    time.sleep(1.5)
    extract_files_from_targz('/tmp/model.tar.gz')
    upload_folder_to_s3(path='/tmp/images', job_name=job_name)


def split_s3_path(s3_path: str) -> str:
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key


def get_url_from_s3_path(s3_path: str) -> str:
    bucket, key = split_s3_path(s3_path)
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


#############################################   STEP 1 #########################################################
# In our case we are converting parquet file to json file so we can use jsonl format later in our training
import pandas as pd
import json

def convert_to_jsonl(input_file, output_file):
    df = pd.read_parquet(input_file)
    df.to_json(output_file, orient='records', lines=True)

input_file1 = r"E:\Cloud_Storage\Stackoverflow\dataset\train-00000-of-00002-de0ae301d048d960.parquet"
output_file1 = r"E:\Cloud_Storage\Stackoverflow\dataset\output1.jsonl"

convert_to_jsonl(input_file1, output_file1)

input_file2 = r"E:\Cloud_Storage\Stackoverflow\dataset\train-00001-of-00002-014071b0381dd5ae.parquet"
output_file2 = r"E:\Cloud_Storage\Stackoverflow\dataset\output2.jsonl"

convert_to_jsonl(input_file2, output_file2)

############################################# STEP 2 ##########################################################

# Now we are merging our dataset so we have large dataset then we divide the dataset into train and eval
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def write_jsonl(df, file_path):
    with open(file_path, 'w') as file:
        for record in df.to_dict(orient='records'):
            file.write(json.dumps(record) + '\n')

def split_data(jsonl_file1, jsonl_file2, train_jsonl, eval_jsonl, test_jsonl, train_size=0.8, eval_size=0.1):
    df1 = read_jsonl(jsonl_file1)
    df2 = read_jsonl(jsonl_file2)

    merged_df = pd.concat([df1, df2], ignore_index=True)
    shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    total_size = len(shuffled_df)
    train_end = int(train_size * total_size)
    eval_end = train_end + int(eval_size * total_size)

    train_df = shuffled_df[:train_end]
    eval_df = shuffled_df[train_end:eval_end]
    test_df = shuffled_df[eval_end:]

    write_jsonl(train_df, train_jsonl)
    write_jsonl(eval_df, eval_jsonl)
    write_jsonl(test_df, test_jsonl)

    print(f"Train data saved to {train_jsonl}")
    print(f"Eval data saved to {eval_jsonl}")
    print(f"Test data saved to {test_jsonl}")

jsonl_file1 = r"E:\Cloud_Storage\Stackoverflow\dataset\output1.jsonl"
jsonl_file2 = r"E:\Cloud_Storage\Stackoverflow\dataset\output2.jsonl"
train_jsonl = r"E:\Cloud_Storage\Stackoverflow\dataset\train.jsonl"
eval_jsonl = r"E:\Cloud_Storage\Stackoverflow\dataset\eval.jsonl"
test_jsonl = r"E:\Cloud_Storage\Stackoverflow\dataset\test.jsonl"

split_data(jsonl_file1, jsonl_file2, train_jsonl, eval_jsonl, test_jsonl)


############################################### STEP 3 ########################################################
#### Push  our dataset to S3 bucket

import boto3
from botocore.exceptions import NoCredentialsError
import os

aws_region = 'us-east-1'  # AWS Region
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
# Created a session using my AWS credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def upload_to_s3(local_file_path, bucket_name, object_name):
    s3 = boto3.client("s3")
    s3.upload_file(local_file_path, bucket_name, object_name)
    print(f"File {local_file_path} uploaded to S3 bucket {bucket_name} as {object_name}")

bucket_name = 'archistackoverflow'  # My S3 bucket name

# Begin to push each file.

#upload_to_s3(jsonl_file1, bucket_name, 'output1.jsonl')
#upload_to_s3(jsonl_file2, bucket_name, 'output2.jsonl')
upload_to_s3(train_jsonl, bucket_name, 'train.jsonl')
upload_to_s3(eval_jsonl, bucket_name, 'eval.jsonl')
upload_to_s3(test_jsonl, bucket_name, 'test.jsonl')


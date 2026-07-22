import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = "us-west-1"
DB_BACKUP_BUCKET = "scuttle-bot-db-backups-314722146857"
RIOT_API_KEY_SECRET_NAME = "scuttle-bot/riot-api-key"

DB_FILES = [
    "src/scuttle_bot/cache/scuttle_bot.db",
    "src/scuttle_bot/cache/ml_dataset.db",
]


def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _secrets_client():
    return boto3.client("secretsmanager", region_name=AWS_REGION)


def backup_databases_to_s3(db_paths: Optional[list] = None, bucket: str = DB_BACKUP_BUCKET) -> list:
    """
    Uploads each local sqlite db file to S3 under its basename. The bucket has
    versioning enabled, so this is safe to run repeatedly (e.g. after a
    training run or on a schedule) without losing prior copies.
    """
    db_paths = db_paths or DB_FILES
    client = _s3_client()
    uploaded = []
    for path in db_paths:
        if not os.path.exists(path):
            logging.warning(f"Skipping backup for {path}: file not found")
            continue
        key = os.path.basename(path)
        client.upload_file(path, bucket, key)
        uploaded.append(key)
    return uploaded


def restore_databases_from_s3(db_paths: Optional[list] = None, bucket: str = DB_BACKUP_BUCKET) -> list:
    """Downloads each db file from S3 back to its local path, overwriting
    whatever is there. Used to bootstrap a fresh machine/instance."""
    db_paths = db_paths or DB_FILES
    client = _s3_client()
    restored = []
    for path in db_paths:
        key = os.path.basename(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            client.download_file(bucket, key, path)
            restored.append(key)
        except ClientError as e:
            logging.warning(f"Could not restore {key} from S3: {e}")
    return restored


def get_riot_api_key(secret_name: str = RIOT_API_KEY_SECRET_NAME) -> Optional[str]:
    """
    Fetches the Riot API key from Secrets Manager. Returns None on any
    failure (no AWS credentials configured, secret not found, network error,
    etc.) so callers can fall back to a local .env value instead of crashing
    -- AWS shouldn't be a hard requirement for local development.
    """
    try:
        response = _secrets_client().get_secret_value(SecretId=secret_name)
        return response["SecretString"]
    except (ClientError, BotoCoreError) as e:
        logging.info(f"Could not fetch {secret_name} from Secrets Manager, falling back to .env: {e}")
        return None


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    command = sys.argv[1] if len(sys.argv) > 1 else "backup"

    if command == "restore":
        restored = restore_databases_from_s3()
        print(f"Restored from s3://{DB_BACKUP_BUCKET}/: {restored}")
    elif command == "backup":
        uploaded = backup_databases_to_s3()
        print(f"Backed up to s3://{DB_BACKUP_BUCKET}/: {uploaded}")
    else:
        print(f"Unknown command {command!r}, expected 'backup' or 'restore'")
        sys.exit(1)

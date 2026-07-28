"""Back up and restore trained model artifacts to/from S3.

Model weights -- especially the RandomForest ``.pkl`` files at ~72 MB each --
are too large to keep in git, and are fully reproducible from ml_dataset.db by
the trainers. They live in S3 instead: this module uploads them after a
training run and restores them onto a fresh checkout. It mirrors the sqlite-db
backup/restore in infra/aws_client.py and reuses the same versioned bucket.

The served model (logistic regression, variant C) and every small artifact
(encoders, scalers, configs, cv summaries) stay tracked in git, so the bot
runs from a clean checkout without an S3 restore; only the heavier auxiliary
RF/NN weights are gitignored and rely on this for recovery. A restore still
pulls *everything* (a full backup), harmlessly overwriting the tracked copies.

CLI (needs AWS credentials, like the db backup). On a dev machine the creds
live in the `scuttle-bot` named profile, which boto3 picks up from AWS_PROFILE;
on EC2 the instance role is used, so no profile is set there:
    AWS_PROFILE=scuttle-bot python -m scuttle_bot.ml.model_store backup   # upload all model artifacts
    AWS_PROFILE=scuttle-bot python -m scuttle_bot.ml.model_store restore  # download them back locally
"""

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from scuttle_bot.infra.aws_client import AWS_REGION, DB_BACKUP_BUCKET

# Reuse the existing versioned backup bucket, namespacing model objects under
# their own prefix so they don't mingle with the db backups at the root.
MODEL_BUCKET = DB_BACKUP_BUCKET
MODEL_PREFIX = "models/"

# Repo-root-relative directories holding trained-model artifacts. Their
# contents are mirrored under MODEL_PREFIX in S3, preserving the path below
# ml/ so a restore lands each file back exactly where it belongs.
ML_ROOT = "src/scuttle_bot/ml"
MODEL_DIRS = [
    f"{ML_ROOT}/rf/models",
    f"{ML_ROOT}/logistic/models",
    f"{ML_ROOT}/nn/models",
]


def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _local_to_key(path: str) -> str:
    """src/scuttle_bot/ml/rf/models/C/rf_model_C_0.pkl
    -> models/rf/models/C/rf_model_C_0.pkl"""
    rel = os.path.relpath(path, ML_ROOT)
    return MODEL_PREFIX + rel.replace(os.sep, "/")


def _key_to_local(key: str) -> str:
    """Inverse of _local_to_key."""
    rel = key[len(MODEL_PREFIX):]
    return os.path.join(ML_ROOT, *rel.split("/"))


def backup_models_to_s3(model_dirs: Optional[list] = None, bucket: str = MODEL_BUCKET) -> list:
    """Uploads every file under each model dir to S3 under MODEL_PREFIX,
    preserving relative paths. Bucket versioning retains prior copies, so this
    is safe to re-run after each training run."""
    model_dirs = model_dirs or MODEL_DIRS
    client = _s3_client()
    uploaded = []
    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            logging.warning(f"Skipping backup for {model_dir}: directory not found")
            continue
        for root, _dirs, files in os.walk(model_dir):
            for name in files:
                path = os.path.join(root, name)
                key = _local_to_key(path)
                client.upload_file(path, bucket, key)
                uploaded.append(key)
    return uploaded


def restore_models_from_s3(bucket: str = MODEL_BUCKET, prefix: str = MODEL_PREFIX) -> list:
    """Downloads every object under the models prefix back to its local path,
    creating directories as needed and overwriting local copies. This is how a
    fresh checkout gets the gitignored auxiliary RF/NN weights back."""
    client = _s3_client()
    restored = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # skip "directory" placeholder keys
            local_path = _key_to_local(key)
            directory = os.path.dirname(local_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            try:
                client.download_file(bucket, key, local_path)
                restored.append(local_path)
            except ClientError as e:
                logging.warning(f"Could not restore {key} from S3: {e}")
    return restored


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    command = sys.argv[1] if len(sys.argv) > 1 else "backup"

    if command == "restore":
        restored = restore_models_from_s3()
        print(f"Restored {len(restored)} model file(s) from s3://{MODEL_BUCKET}/{MODEL_PREFIX}")
    elif command == "backup":
        uploaded = backup_models_to_s3()
        print(f"Backed up {len(uploaded)} model file(s) to s3://{MODEL_BUCKET}/{MODEL_PREFIX}")
    else:
        print(f"Unknown command {command!r}, expected 'backup' or 'restore'")
        sys.exit(1)

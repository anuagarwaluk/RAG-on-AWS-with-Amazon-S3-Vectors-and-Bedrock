"""Tear down demo resources: indexes first, then the vector bucket.

Explicit confirmation is required: deleting a vector index deletes every
vector in it, and there is no undo.
"""

from botocore.exceptions import ClientError

from . import config
from .aws_clients import client_error_summary, s3vectors_client


def cleanup_all(
    bucket_name: str = config.VECTOR_BUCKET_NAME,
    confirm: bool = False,
) -> None:
    if not confirm:
        print("Cleanup not run. Re-run with --confirm to delete all demo resources.")
        return
    s3v = s3vectors_client()
    try:
        indexes = s3v.list_indexes(vectorBucketName=bucket_name).get("indexes", [])
        for idx in indexes:
            name = idx["indexName"]
            s3v.delete_index(vectorBucketName=bucket_name, indexName=name)
            print(f"Deleted index: {name}")
        s3v.delete_vector_bucket(vectorBucketName=bucket_name)
        print(f"Deleted vector bucket: {bucket_name}")
    except ClientError as exc:
        print(f"Cleanup issue (may already be deleted): {client_error_summary(exc)}")

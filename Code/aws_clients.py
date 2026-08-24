"""Boto3 clients and shared error helpers.

S3 Vectors requires boto3 >= 1.42.0 (the `s3vectors` client). If the client
cannot be created, upgrade the SDK before debugging anything else.
"""

from functools import lru_cache
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from . import config


@lru_cache(maxsize=1)
def s3vectors_client():
    return boto3.client("s3vectors", region_name=config.AWS_REGION)


@lru_cache(maxsize=1)
def bedrock_runtime_client():
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION)


def client_error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def client_error_summary(exc: ClientError) -> str:
    err = exc.response.get("Error", {})
    return f"{err.get('Code', 'Unknown')}: {err.get('Message', str(exc))}"


def as_float32_list(values: List[float], expected_dimensions: Optional[int] = None) -> List[float]:
    """Coerce an embedding to a plain float list and validate its dimension.

    S3 Vectors stores float32 only; a dimension mismatch at query time is the
    classic symptom of using a different embedding model or dimension setting
    than the one used at index time, so fail loudly here.
    """
    floats = [float(v) for v in values]
    if expected_dimensions is not None and len(floats) != expected_dimensions:
        raise ValueError(
            f"Embedding has {len(floats)} dimensions, expected {expected_dimensions}. "
            "Index-time and query-time embedding settings must match exactly."
        )
    return floats

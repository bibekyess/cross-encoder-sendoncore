# download mlflow model
import logging
from os import environ
from pathlib import Path

from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from mlflow import MlflowException
from mlflow import artifacts as mlflow_artifacts

logger = logging.getLogger(__name__)


def _download_model_artifact(artifact_uri: str, path: Path) -> None:
    """Get the model URI from the tracking server and download it.

    Args:
        artifact_uri (str): The uri for the artifact to download
        path (Path): Where to download the model to

    Raises:
        ValueError: Download failed because of a known configuration issue.
        MlflowException: Download failed because of an unhandled MLflow error.
    """
    try:
        mlflow_artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=f"{path}/model_from_mlflow")
    except MlflowException as e:
        if not environ.get("MLFLOW_TRACKING_URI"):
            raise ValueError(
                "Could not reach the MLflow tracking server because MLFLOW_TRACKING_URI was not set."
            ) from e
        raise e
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise ValueError(
            "Tried to download artifact from AWS S3, but found no complete credentials set."
        ) from e
    except ClientError as e:
        raise ValueError(
            "The AWS SDK could not reach the resource because of client configuration errors."
        ) from e
    logger.debug("Downloaded model %s to %s", artifact_uri, path)


_download_model_artifact(artifact_uri="s3://mlflow/0/d3ac0e6fa8824a2980b8041c05612b66/artifacts/cross_encoder_pyfunc", path='./')
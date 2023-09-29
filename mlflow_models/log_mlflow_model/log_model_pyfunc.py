from typing import Dict, Any, Optional
import logging
import numpy as np
import mlflow
import sentence_transformers
import cloudpickle
from sys import version_info
import pandas as pd

# mlflow must be running with database backend
mlflow.set_tracking_uri('<YOUR-VALUE>') # FIXME

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# This is the path of theh huggingface downloaded model
artifacts = {"model_path": "../../huggingface_models/mlserver_image_build/model/ms-marco-MiniLM-L-6-v2"}

class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Dict[str, Any]) -> bool:
        from sentence_transformers import CrossEncoder
        
        # Load SentenceTransformer from disk.
        self.model = CrossEncoder(context.artifacts["model_path"])
        self.ready = True
        return self.ready
    
    def predict(self, context: Dict[str, Any], model_input: pd.DataFrame,  params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        print(model_input)

        inputs = model_input.get("inputs")
        # InferenceRequest is not automatically decoded
        if inputs:
            model_input_ = {}
            for inp in inputs:
                model_input_[inp['name']] = inp['data']
            model_input = model_input_

        query = model_input["query"]
        paragraphs = model_input["paragraphs"]
        input = []
        for p in paragraphs:
            input.append([query[0], p])  
        scores = self.model.predict(input)
        logger.info(scores)
        return scores

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        f"python={PYTHON_VERSION}",
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"sentence-transformers=={sentence_transformers.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                f"numpy=={np.__version__}"
            ],
        },
    ],
    "name": "cross_encoder_env",
}

# Save the MLflow Model
mlflow_pyfunc_model_path = "cross_encoder_pyfunc"
mlflow.pyfunc.log_model(
    artifact_path = mlflow_pyfunc_model_path,
    python_model=CrossEncoderWrapper(),
    artifacts=artifacts,
    conda_env=conda_env
)

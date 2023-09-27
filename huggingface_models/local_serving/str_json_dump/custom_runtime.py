from mlserver import MLModel, types
from mlserver.utils import get_model_uri
from mlserver.codecs import StringCodec
from sentence_transformers import CrossEncoder
from typing import Dict, Any
import logging
import torch
import json


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CrossEncoderRuntime(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SentenceTransformer from disk.
        self.model = CrossEncoder(model_uri, device=self.device)
        self.ready = True
        return self.ready
    
    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        request = self._extract_json(payload).get("inference_request", {})
        query = request.get("query", '')
        paragraphs = request.get("paragraph", [])
        # logger.info(f"Query: {query}")
        # logger.info(f"Paragraphs: {paragraphs}")

        input = []
        for p in paragraphs:
            input.append([query, p])  

        scores = self.model.predict(input)
        # logger.info(scores)
        output_data = {"success": True}
        output_data["scores"] = scores.tolist()
        response_bytes = json.dumps(output_data).encode("UTF-8")

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="model_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=types.Parameters(content_type="str"),
                )
            ],
        )
    
    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            # logger.info(inp)
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )
        
        return inputs

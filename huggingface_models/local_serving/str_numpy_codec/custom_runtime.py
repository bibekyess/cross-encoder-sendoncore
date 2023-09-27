from mlserver import MLModel, types
from mlserver.utils import get_model_uri
from mlserver.codecs import StringCodec, NumpyCodec
from sentence_transformers import CrossEncoder
from typing import Dict, Any
import logging
import torch


# Configure logging
logging.basicConfig(level=logging.DEBUG) #FIXME
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
        request = self._extract_json(payload)
        query = request.get("query", [])
        paragraphs = request.get("paragraphs", [])
        # logger.info(f"Query: {query}")
        # logger.info(f"Paragraphs: {paragraphs}")

        input = []
        for p in paragraphs:
            input.append([query[0], p])  

        scores = self.model.predict(input)

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs= [NumpyCodec.encode_output(name='model_response', payload = scores, use_bytes=False)]
        )
    
    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = StringCodec.decode_input(inp)
        return inputs

from mlserver import MLModel
from mlserver.utils import get_model_uri
from sentence_transformers import CrossEncoder
from typing import List
import logging
import numpy as np
from mlserver.codecs import decode_args

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CrossEncoderRuntime(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        
        # Load SentenceTransformer from disk.
        self.model = CrossEncoder(model_uri)
        self.ready = True
        return self.ready
    
    @decode_args
    async def predict(self, query: List[str], paragraphs: List[str]) -> np.ndarray:
        # logger.info(f"Query: {query}")
        # logger.info(f"Paragraphs: {paragraphs}")
        input = []
        for p in paragraphs:
            input.append([query[0], p])  
        scores = self.model.predict(input)
        # logger.info(scores)
        return scores

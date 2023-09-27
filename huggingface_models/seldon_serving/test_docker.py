# Import required dependencies
import requests
from mlserver.types import InferenceResponse
from mlserver.codecs import StringCodec, NumpyCodec

inputs = {
            "query": ["Query"],
            "paragraphs": ["Paragraph1", "Paragraph2", "Paragraph3"]
         }

inference_request = {
    "inputs": [
        StringCodec.encode_input(name='query', payload=inputs["query"], use_bytes=False).dict(),
        StringCodec.encode_input(name='paragraphs', payload=inputs["paragraphs"], use_bytes=False).dict()
    ]
}
# print(inference_request)

inference_url = "http://localhost:8080/v2/models/cross-encoder/infer"

response = requests.post(inference_url, json=inference_request)
# print(response.json())
inference_response = InferenceResponse.parse_raw(response.text)
output_scores = NumpyCodec.decode_output(inference_response.outputs[0])
print(output_scores)

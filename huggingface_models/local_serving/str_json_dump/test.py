# Import required dependencies
import requests
import json
import json
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec

inputs = {
            "query": "Query",
            "paragraph": ["Paragraph1", "Paragraph2", "Paragraph3"]
         }
inputs_string = json.dumps(inputs)

inference_request = {
    "inputs": [
        {
          "name": "inference_request",
          "shape": [len(inputs_string)],
          "datatype": "BYTES",
          "data": [inputs_string],
        }
    ]
}

inference_url = "http://localhost:8080/v2/models/cross-encoder/infer"

response = requests.post(inference_url, json=inference_request)
# print(response.json())
inference_response = InferenceResponse.parse_raw(response.text)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
print(output)

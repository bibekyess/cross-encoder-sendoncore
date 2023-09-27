# Import required dependencies
import requests
from mlserver.types import InferenceResponse
from mlserver.codecs import StringCodec, NumpyCodec
import os

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

inference_url = "http://localhost:8082/seldon/seldon/mlserver-cross-encoder/v2/models/cross-encoder/infer"

# You can copy the session_cookie from the web browser and use an environment variable to use it
# SESSION_COOKIE = os.getenv('SESSION_COOKIE')

# Or
USERNAME = "user@example.com"
PASSWORD = "12341234" 
NAMESPACE = "kubeflow-user-example-com"
HOST = "http://localhost:8082" # your istio-ingressgateway-pod-ip:8082

session = requests.Session()
response = session.get(HOST)

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {"login": USERNAME, "password": PASSWORD}
session.post(response.url, headers=headers, data=data)
SESSION_COOKIE = session.cookies.get_dict()["authservice_session"]


headers = {"accept": "application/json", "Content-Type": "application/json", "Cookie": f"authservice_session={SESSION_COOKIE}"}

response = requests.post(url=inference_url, headers=headers, json=inference_request)
# print(response.text)
inference_response = InferenceResponse.parse_raw(response.text)
output_scores = NumpyCodec.decode_output(inference_response.outputs[0])
print(output_scores)

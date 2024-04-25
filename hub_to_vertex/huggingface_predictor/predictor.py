import os
import logging
import tarfile
from typing import Any, Dict
import torch

from transformers import AutoModel, AutoTokenizer, pipeline

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HuggingFacePredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None

    def load(self, artifacts_uri: str) -> None:
        """Loads the preprocessor and model artifacts."""
        # logger.info(f"Downloading artifacts from {artifacts_uri}")
        prediction_utils.download_model_artifacts(artifacts_uri)
        # logger.info("Artifacts successfully downloaded!")
        os.makedirs("./model", exist_ok=True)
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(path="./model")
        # logger.info(f"HF_TASK value is {os.getenv('HF_TASK', 'default')}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.model = AutoModel.from_pretrained("./model")

        logger.info("`Model` and `Tokenizer` successfully loaded!")

    def predict(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        # Convert texts to model inputs
        logger.info(f"Received instances: {instances}")
        # try:
        #     logger.info(f"Received instances.instances: {instances['instances']}")
        # except:
        #     logger.info("not correct format")

        # encoded_input = self.tokenizer(instances["instances"][0]["sequences"], return_tensors="pt", padding=True, truncation=True)
        encoded_input = self.tokenizer(instances["sequences"], return_tensors="pt", padding=True, truncation=True)
        logger.info(f"Encoded input: {encoded_input}")
        with torch.no_grad():
            # model_output = self.model(**encoded_input, attention_mask=encoded_input["attention_mask"])
            model_output = self.model(**encoded_input)
            logger.info(f"Model output: {model_output}")

        results = mean_pooling(model_output, encoded_input['attention_mask'])
        # logger.info(f"After Mean Pooling: {results}")
        # logger.info(f"Pooler output: {model_output.pooler_output}")
        # # also log the size 
        # logger.info(f"Pooler output size: {model_output.pooler_output.shape}")
        # logger.info(f"Results size: {results.shape}")
        logger.info(f"Before: {results}")
        results = results.tolist()
        logger.info(f"After: {results}")
        logger.info(f"Results type: {type(results)}")
        return {"Model Output" : results}
        
        # return {"results": results}

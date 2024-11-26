import os
import torch
from litserve import LitServer
from huggingface_hub import login
from models import * 

login(token="***********************")
HF_MODEL_NAME = os.environ["HF_MODEL_NAME"]


if __name__ == "__main__":

    if HF_MODEL_NAME == "annyalvarez/Roberta-MultiMoral-Polarity-MS-DM-P2":
        api = RobertaMultiMoralPolarityAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "annyalvarez/Roberta-MultiMoralPres-MS-P1":
        api = RobertaMultiMoralPresenceAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "annyalvarez/Roberta-MoralPres-MS-DM-P2":
        api = RobertaMoralPresenceAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "annyalvarez/Roberta-MoralPolarity-MS-P1":
        api = RobertaMoralPolarityAPI(HF_MODEL_NAME)
    else:
        raise ValueError(f"HF_MODEL_NAME env is not set properly. Its value is {HF_MODEL_NAME}")
    
    server = LitServer(api, accelerator='cuda' if torch.cuda.is_available() else 'cpu', devices=1)
    
    server.run(port=8000)


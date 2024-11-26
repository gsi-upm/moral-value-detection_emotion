import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

model_urls = {
    "moralpolarity_model": "http://moralpolarity_model:8000/predict",
    "moral_model": "http://moral_model:8000/predict",
    "multimoralpolarity_model": "http://multimoralpolarity_model:8000/predict",
    "multimoral_model": "http://multimoral_model:8000/predict"}


#request
class RequestData(BaseModel):
    text: str
    model_name: str


@app.post("/predict")
async def predict(request_data: RequestData):
    text = request_data.text
    model_name = request_data.model_name

    if model_name not in model_urls:
        raise HTTPException(
            status_code=400,
            detail="Model name not valid. Available models: moralpolarity_model, moral_model, multimoralpolarity_model, multimoral_model."
        )
    url = model_urls[model_name]
    response = requests.post(url, json={"text": text})

    return response.json()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

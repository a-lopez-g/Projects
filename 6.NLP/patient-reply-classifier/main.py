from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.controllers.make_request_model import RequestModel
from src.controllers.detect_out_of_place_opens import (
    detect_out_of_place_opens_controller,
)


app = FastAPI(
    title="Tucuvi Detect Out-Of-Place Open Responses API",
    description="Made by Andrea",
    version="0.0.1",
    contact={
        "name": "S.L",
        "url": "https://",
        "email": "info@tucuvi.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.post("/", tags=["API"])
def detect_out_of_place_open_responses(request: RequestModel):
    request_json = request.dict()
    results = detect_out_of_place_opens_controller(request_json)
    return results


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"],
)

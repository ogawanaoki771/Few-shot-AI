from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from PIL import Image
import numpy as np
import io
import os


from train import train_model
from predict import predict_image



app = FastAPI()



os.makedirs(
    "data/train",
    exist_ok=True
)

os.makedirs(
    "data/test",
    exist_ok=True
)



@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )



# ==========================
# PNG → 64×64 numpy
# ==========================

@app.post("/upload")
async def upload(
    file:UploadFile=File(...)
):


    data = await file.read()


    img = Image.open(
        io.BytesIO(data)
    ).convert("L")


    img = img.resize(
        (64,64)
    )


    arr=np.array(
        img
    )


    arr = (
        arr < 128
    ).astype(
        np.float32
    )



    filename = (
        "data/train/"
        +
        file.filename
        +
        ".npy"
    )


    np.save(
        filename,
        arr
    )


    return {

        "saved":filename,

        "shape":
        arr.shape

    }



# ==========================
# train
# ==========================


@app.post("/train")
def train():

    result=train_model()


    return result



# ==========================
# predict
# ==========================


@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):


    data=await file.read()


    result=predict_image(
        data
    )


    return result
    }

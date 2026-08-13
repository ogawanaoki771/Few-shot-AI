from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import io
import os


app = FastAPI()


os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)



@app.get("/")
def home():
    return FileResponse("index.html")



@app.post("/upload")
async def upload(
    file: UploadFile = File(...)
):

    img_bytes = await file.read()

    img = Image.open(
        io.BytesIO(img_bytes)
    ).convert("L")


    # 50×50へ縮小
    img = img.resize(
        (50,50)
    )


    arr = np.array(img)


    # 二値化
    arr = (arr < 128).astype(np.float32)


    filename = "data/train/img_001.npy"


    np.save(
        filename,
        arr
    )


    return {
        "message":"saved",
        "shape":arr.shape
    }

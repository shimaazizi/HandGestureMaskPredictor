from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO
import io
from PIL import Image as im 

app = FastAPI()

# Load model 
model = torch.load("/home/shima98/HandGestureMaskPredictor/unet_model.pth")


@app.post("/predict")
def get_segmentation_map(file: bytes =  File(...)):
    """Get segmentation maps from image file"""
    image = Image.open(io.BytesIO(file))
    np_image = np.array(image)
    segmented_np_image = model(np_image)
    output_image = im.fromarray(segmented_np_image)


    bytes_io = io.BytesIO()
    output_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")






























import io 
from segmentation import segmentator, get_segments
from starlette.responses import Response

from fastapi import FastAPI, File

import uvicorn

model = segmentator()


app = FastAPI(
    title="predict mask"
)

@app.post("/image segmentation")
async def get_segmentation(file: bytes = File(...)):
        """Get segmentation maps from image file"""
        segmented_image = get_segments(model, file)
        bytes_io = io.BytesIO()
        segmented_image.save(bytes_io, format="PNG")
        return Response(bytes_io.getvalue(), media_type="image/png")


import io                                   
import os                                   
import numpy as np                          
import tensorflow as tf                    
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse  
from PIL import Image, ImageOps, ImageFilter  

app = FastAPI(title="Digit Classifier API")

# load the model that train.py saved
# (if you forgot to run train.py, this line will give an error)
model = tf.keras.models.load_model("digit_model.keras")

# folder where this file is kept (so we can find index.html)
folder = os.path.dirname(os.path.abspath(__file__))


# home route - shows the simple web page
@app.get("/")
def home():
    return FileResponse(os.path.join(folder, "index.html"))


# this function makes ANY image look like an MNIST image
# MNIST digit = white digit, black background, digit fits in a 20x20 box, centered in 28x28
def prepare_image(img):
    # if the image has transparency (like a PNG from the canvas), put it on a white background
    img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, "white")
    white.alpha_composite(img)

    # convert to grayscale (L = one channel, black and white)
    img = white.convert("L")

    # MNIST digits are WHITE on a BLACK background.
    # If most of the image is bright (black digit on white paper), flip the colors.
    if np.median(np.array(img)) > 127:
        img = ImageOps.invert(img)

    # find the box that contains the digit (pixels brighter than 50)
    box = img.point(lambda p: 255 if p > 50 else 0).getbbox()
    if box is None:
        return None          # nothing was drawn in the image

    # crop the image to only the digit (removes the empty space around it)
    img = img.crop(box)

    # make the cropped image a square by pasting it in the middle of a black square
    w, h = img.size
    side = max(w, h)
    square = Image.new("L", (side, side), 0)
    square.paste(img, ((side - w) // 2, (side - h) // 2))

    # make the pen stroke thicker, because MNIST digits have thick strokes
    # (the bigger the digit, the bigger the thickening)
    k = (side // 20) * 2 + 1     # must be an odd number
    if k < 3:
        k = 3
    square = square.filter(ImageFilter.MaxFilter(k))

    # shrink the digit to 20x20 and put it in the center of a 28x28 black image
    square = square.resize((20, 20), Image.LANCZOS)
    final = Image.new("L", (28, 28), 0)
    final.paste(square, (4, 4))
    return final


# prediction route - we send an image file here using POST
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read the uploaded file as bytes
    contents = await file.read()

    # try to open it as an image, if it is not an image show an error
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    # clean the image so it looks like the training images
    img = prepare_image(img)
    if img is None:
        raise HTTPException(status_code=400, detail="No digit found in the image")

    # convert to numpy array and scale pixels to 0-1 (same as training)
    arr = np.array(img).astype("float32") / 255.0

    # reshape to (1 image, 28, 28, 1 channel) because model expects a batch
    arr = arr.reshape(1, 28, 28, 1)

    # ask the model for probabilities of all 10 digits
    probs = model.predict(arr, verbose=0)[0]

    # the digit with the highest probability is our answer
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # probability of every digit 0-9 (the web page uses this for the bar chart)
    all_probs = [round(float(p), 4) for p in probs]

    return {
        "predicted_digit": digit,
        "confidence": round(confidence, 4),
        "probabilities": all_probs,
    }

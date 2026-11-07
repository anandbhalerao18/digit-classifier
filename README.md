# Handwritten Digit Classifier

A deep learning project that recognises handwritten digits (0-9). A Convolutional Neural Network (CNN) is trained on the **MNIST** dataset with TensorFlow/Keras, saved to a file, and served through a **FastAPI** REST API with a simple web page for testing.

## Features

- CNN trained on 60,000 MNIST images (about 99% test accuracy)
- Trained model saved as `digit_model.keras`
- REST API with a `POST /predict` endpoint
- Automatic image cleaning: any digit image is converted to MNIST style (grayscale, inverted if needed, cropped, centered, thickened, resized to 28x28)
- Web page with drag-and-drop upload, image preview, predicted digit, confidence and a probability bar chart for all 10 digits
- Auto-generated API docs at `/docs`
- Demo images in the `testing demos` folder to try the project quickly

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Build, train, save and load the CNN |
| NumPy | Array handling |
| FastAPI | REST API |
| Uvicorn | Server that runs the API |
| python-multipart | Receiving uploaded files |
| Pillow | Image processing |
| Requests | Testing the API from a script |

## Project Structure

```
digit_project/
├── train.py            # trains the CNN and saves digit_model.keras
├── app.py              # FastAPI app (prediction API + serves the web page)
├── index.html          # web page frontend
├── test_api.py         # test the API from the terminal
├── requirements.txt    # Python libraries needed
├── testing demos/      # sample digit images for testing (demo1, demo2, ...)
└── digit_model.keras   # created after running train.py
```

## Setup and Run

Python 3.9 to 3.12 is recommended, because TensorFlow may not support the newest Python versions.

**1. Open a terminal inside the project folder**

```
cd digit_project
```

**2. (Optional) create a virtual environment**

```
python -m venv venv
venv\Scripts\activate
```

On Mac/Linux use `source venv/bin/activate` for the second command.

**3. Install the libraries**

```
python -m pip install -r requirements.txt
```

**4. Train and save the model (only needed once)**

```
python train.py
```

This downloads MNIST the first time, trains for 5 epochs and saves `digit_model.keras`. It takes a few minutes at most.

**5. Start the API server**

```
python -m uvicorn app:app --reload
```

Wait for `Application startup complete.`

**6. Open the app**

Go to http://127.0.0.1:8000 in your browser, upload a digit image and click **Predict Digit**.

To stop the server press `Ctrl + C`.

## Testing with the Demo Images

The `testing demos` folder contains sample digit images (demo1, demo2, ...). To try them:

- **In the browser:** open http://127.0.0.1:8000, click the upload box and pick any image from `testing demos`.
- **From the terminal** (start the server first, then open a second terminal in the project folder). Put the path in quotes because the folder name has a space:

```
python test_api.py "testing demos/demo1.png"
```

Change `demo1.png` to the real file name and extension of your image.

- **From the API docs page:** open http://127.0.0.1:8000/docs, choose `POST /predict`, click **Try it out**, upload an image and click **Execute**.

## API Reference

### `GET /`

Returns the web page.

### `POST /predict`

Send an image file as `multipart/form-data` with the field name `file`.

Example with curl:

```
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@digit.png"
```

Example response:

```json
{
  "predicted_digit": 8,
  "confidence": 0.9731,
  "probabilities": [0.0, 0.0, 0.0114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9731, 0.0155]
}
```

- `predicted_digit`: the digit the model thinks it is
- `confidence`: probability of that digit (0 to 1)
- `probabilities`: probability of each digit from 0 to 9

Errors return status `400` with a message, for example when the file is not an image or no digit is found.

## How It Works

**Model (`train.py`)**

```
Input 28x28x1
-> Conv2D(32, 3x3, ReLU) -> MaxPooling(2x2)
-> Conv2D(64, 3x3, ReLU) -> MaxPooling(2x2)
-> Flatten
-> Dense(64, ReLU)
-> Dense(10, Softmax)
```

Pixels are scaled to 0-1, then the model is trained with the Adam optimizer and sparse categorical crossentropy loss for 5 epochs.

**API (`app.py`)**

1. Load the saved model once when the server starts.
2. Receive an uploaded image on `/predict`.
3. Clean it to look like an MNIST image (grayscale, invert if the background is white, crop to the digit, center it, thicken the stroke, resize to 28x28).
4. Scale to 0-1 and reshape to `(1, 28, 28, 1)`.
5. Run `model.predict()` to get 10 probabilities.
6. Return the digit with the highest probability as JSON.

## Tips for Good Predictions

- Use one clear digit per image.
- A plain background (white paper or black) works best.
- A thick pen stroke works better than a very thin line.

## Limitations

- Works for single digits only.
- Very unusual handwriting can be predicted wrongly, because the model learned only from MNIST style digits.
- Photos with messy backgrounds reduce accuracy.

## Future Improvements

- Add Dropout and data augmentation (small rotations and shifts)
- Train for more epochs or use a deeper CNN
- Add a drawing canvas to the web page
- Recognise multiple digits in one image
- Deploy with Docker on a cloud service

## Troubleshooting

| Problem | Fix |
|---|---|
| `No such file digit_model.keras` | Run `python train.py` first |
| `uvicorn` is not recognized | Use `python -m uvicorn app:app --reload` |
| `Form data requires python-multipart` | Run `python -m pip install python-multipart` |
| Could not reach the server | Make sure the server window is still running |
| TensorFlow install or import errors | Use Python 3.12 or older |
| Port 8000 already in use | Close the old server window or press `Ctrl + C` in it |

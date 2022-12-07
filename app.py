from fastapi import FastAPI, File
from keras.models import load_model
import numpy as np

classes=["a clear sky","a cloudy sky","a partly cloudy sky","a rain cloud"]
filename="model_m.h5"
app = FastAPI()

app.state.model = load_model(filename)

@app.get("/")
def root():
    return {"Hello this is an API"}

@app.post("/predict")
async def predict(bytes: bytes=File(...)):

    img = np.array(np.frombuffer(bytes, dtype=np.uint8))/255 # Convert from byte to array
    img= img.reshape((-1, 224, 224, 3)) # Reshape
    res = app.state.model.predict(img) # Predict

    if filename=="model_b.h5":

        res = app.state.model.predict(img)[0][0]

        if(res < 0.5):
            weather = "no rain cloud"
            prob = 1-res
            prob = "{:.0%}".format(prob)
        if(res >= 0.5):
            weather = "a rain cloud"
            prob = res
            prob = "{:.0%}".format(prob)

        print("Weather : ", weather)
        print("probability = ",prob)

    if filename=="model_m.h5":

        res = app.state.model.predict(img)
        weather = classes[np.argmax(res[0])]
        prob = np.max(res[0])
        prob = "{:.0%}".format(prob)
        if prob=="100%":
            prob="99%"

        print("Weather : ", weather)
        print("probability = ",prob)

    result = f"This part of the image shows {weather}. The model has calculated a probability of {prob}."
    return result

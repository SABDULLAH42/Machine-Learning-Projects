from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

model = joblib.load("student_dropout_model.pkl")
features = joblib.load("model_features.pkl")

app = FastAPI(
    title = "Student Dropout Production API",
    description = "Predict whether a student will drop out (1) or stay active (0)",
    version = "1.0"
)

class StudentData(BaseModel):
    completion_rate: float = Field(..., example = 0.45)
    login_frequency: float = Field(..., example = 3.2)
    last_activity_days_ago: int = Field(..., example = 12)
    courses_enrolled: int = Field(..., example = 4)
    forum_posts_count: int = Field(..., example =2)


@app.get("/")
def home():
    return{"message": "Student Dropout Prediction API is ready."}

@app.post("/predict")
def predict(data: StudentData):

    input_data = np.array([[
        data.completion_rate,
        data.login_frequency,
        data.last_activity_days_ago,
        data.courses_enrolled,
        data.forum_posts_count
    ]])

    prediction = model.predict(input_data)[0]

    return{"prediction" : int(prediction)}


# http://127.0.0.1:8000/docs

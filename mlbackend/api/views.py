import joblib
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response

model = joblib.load("api/house_model.pkl")

@api_view(["POST"])
def predict_price(request):
    area = float(request.data.get("area", 0))
    bedrooms = float(request.data.get("bedrooms", 0))

    prediction = model.predict([[area, bedrooms]])[0]
    return Response({"predicted_price": round(prediction, 2)})
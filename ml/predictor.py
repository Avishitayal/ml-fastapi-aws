import joblib
import numpy as np

# Load trained model
model = joblib.load("ml/model.pkl")

def predict_house_price(features):
    """
    features must contain 8 values:
    [
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]
    """

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)

    return float(prediction[0])
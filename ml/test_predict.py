from predictor import predict_house_price

sample_data = [
    8.3252,
    41,
    6.984127,
    1.02381,
    322,
    2.555556,
    37.88,
    -122.23
]

prediction = predict_house_price(sample_data)

print("Predicted House Price:", prediction)
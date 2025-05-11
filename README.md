# 🚖 Taxi Trip Demand Prediction

This project predicts the number of taxi trips for a given pickup time and location in NYC.  
It uses a machine learning model (`RandomForestRegressor`) trained on historical data, and serves predictions through a FastAPI web service. The application is containerized using Docker for easy deployment.


## 📦 Features

- Preprocessed input data
- Model training pipeline with MLflow tracking
- Random Forest model for regression
- FastAPI service for predictions
- Docker support for easy deployment
- Logging and configuration separation


## 🚀 Getting Started

### 🐳 Run the project using Docker

To build and start the application:

```bash
docker build -t taxi-demand .
docker run -p 8080:8080 taxi-demand
```

## ✅ Example CURL Command
```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pickup_hour": 23,
  "pickup_day": 31,
  "pickup_weekday": 6,
  "is_weekend": 1,
  "PUlocationID": 1
}'
```
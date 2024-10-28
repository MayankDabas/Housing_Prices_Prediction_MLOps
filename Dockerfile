# Dockerfile

# ==============================================================================
# Dockerfile for House Price Prediction Model
# ------------------------------------------------------------------------------
# This Dockerfile creates a container for serving predictions from a trained 
# RandomForest model using a Flask-based API. It performs the following steps:
#
# 1. Uses a lightweight base image (python:3.9-slim) to minimize image size.
# 2. Sets the working directory to /app.
# 3. Copies and installs the dependencies specified in requirements.txt.
# 4. Installs Flask to serve the model via a REST API.
# 5. Copies the entire project directory to the container.
# 6. Sets the command to run the Flask-based inference API.
# ==============================================================================


# Uncomment this to run house_price_prediction.py for hyperparameter tuning
# FROM python:3.9-slim
# WORKDIR /app
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# COPY . .
# CMD ["python", "house_price_prediction.py"]


FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install flask
COPY . .
CMD ["python", "inference_api.py"]


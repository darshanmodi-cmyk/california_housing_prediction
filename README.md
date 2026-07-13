# 🏡 California Housing Price Prediction

A Machine Learning-powered web application that predicts California house prices using a **Random Forest Regression** model. The project provides a **FastAPI backend** for predictions and a simple, interactive frontend built with **HTML, CSS, and JavaScript**.

---

## 🚀 Features

* Predict California house prices in real time
* FastAPI REST API
* Random Forest Regression model
* Interactive and responsive frontend
* Input validation using Pydantic
* CORS enabled for frontend integration
* Automatic API documentation with Swagger UI

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* HTML
* CSS
* JavaScript

---


## 📊 Dataset

This project uses the **California Housing Dataset** provided by Scikit-learn.

Features used for prediction:

* Median Income (MedInc)
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/california_housing_prediction.git
cd california_housing_prediction
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app:app --reload
```

## 📚 API Documentation

FastAPI automatically generates interactive API documentation.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## 🎯 Future Improvements

* Deploy the API to Render or Railway
* Improve model accuracy with hyperparameter tuning
* Add interactive charts and data visualizations
* Support batch predictions using CSV uploads
* Implement user authentication
* Containerize the application with Docker

---

## 👨‍💻 Author

Darshan Modi

If you found this project helpful, feel free to ⭐ the repository and share your feedback!

✈️ Flight Price Prediction
A machine learning project for predicting airline ticket prices using a Random Forest Regressor.

📌 Project Overview

This project predicts flight ticket prices based on:

Airline
Source city
Destination city
Departure time
Arrival time
Number of stops
Flight class
Flight duration
Days left before departure
The project includes:

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data preprocessing
Exploratory Data Analysis (EDA)
Model training
Hyperparameter tuning
FastAPI backend
Interactive HTML dashboard
📂 Project Structure
├── api.py
├── flight.ipynb
├── flight-fare-dashboard.html
├── Clean_Dataset.csv
├── encoder.pkl
├── scaler.pkl
├── columns.pkl
├── .gitignore
└── README.md
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🤖 Trained Model
The trained model (final_model.pkl) is not included in this repository because it exceeds GitHub's file size limit.

Download it from Google Drive: https://drive.google.com/file/d/1TOlp7hE1HxxUyS1fMzvJqCaJ4sGTjXI4/view?usp=drive_link

After downloading:

Extract the ZIP file (if needed).
Place final_model.pkl in the project root directory (same folder as api.py).
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Run the API
Install the required packages:

pip install fastapi uvicorn pandas numpy scikit-learn joblib
Run the server:

uvicorn api:app --reload
The API will be available at:

http://127.0.0.1:8000
Swagger documentation:

http://127.0.0.1:8000/docs
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🌐 Dashboard
Open:

flight-fare-dashboard.html
Enable Use Real Model and keep the API running to obtain predictions from the trained model.

🛠 Technologies
Python
Pandas
NumPy
Scikit-learn
FastAPI
HTML
CSS
JavaScript
Chart.js
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
👥 The Team
| Farha Mohamed (Lead) | @farha-mohamed | LinkedIn |

| Haneen Sobhy | @haneensobhey | LinkedIn |

| Nora Salama | @nora12hamed34salama56 | LinkedIn |

| Hajar Alaa | @hajar3laa | LinkedIn |

| Taha Mohammed | @Taha-M-H | LinkedIn |

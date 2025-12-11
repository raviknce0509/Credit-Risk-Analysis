# Credit Risk Prediction API 🏦

## 📌 Project Overview
Financial institutions lose millions annually due to loan defaults. This project is an **End-to-End Machine Learning Pipeline** designed to predict the likelihood of a borrower defaulting on a loan.

## 🚀 Key Features
* **Machine Learning:** Trained a Random Forest model achieving **82% Accuracy**.
* **Production Ready:** Model serialized using `joblib` and served via **FastAPI**.
* **Containerized:** Dockerized for consistent deployment across any environment.
* **Business Impact:** capable of reducing risk exposure by identifying high-risk applicants in real-time.

## 🛠️ Tech Stack
* **Python** (Pandas, Scikit-Learn)
* **API Framework:** FastAPI
* **Deployment:** Docker
* **Cloud:** AWS (EC2/ECS) - *[Add this after we deploy!]*

## 📊 How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `uvicorn app:app --reload`
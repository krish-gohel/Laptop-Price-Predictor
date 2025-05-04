# 💻 Laptop Price Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red?logo=streamlit)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)]()

A machine learning-powered web application that predicts the price of a laptop based on its specifications. Users can enter key hardware details, and the app will estimate the laptop's price using a trained regression model.

---

## 🚀 Features

- ✅ Real-time price prediction
- ✅ Interactive form to input laptop specs
- ✅ Data preprocessing and encoding
- ✅ Clean and responsive web UI
- ✅ Deployed using Streamlit Cloud

---

## 🧠 Technologies Used

| Category     | Tools/Libraries |
|--------------|------------------|
| Frontend     | Streamlit (or Flask + HTML/CSS) |
| Backend/ML   | Python, Pandas, NumPy, Scikit-learn |
| Model        | Random Forest / Linear Regression |
| Deployment   | Streamlit Cloud / Render |

---

## 📊 Model Details

- Trained on a dataset of real-world laptop specifications and prices.
- Features include: brand, processor, RAM, storage, GPU, screen type, etc.
- Regression algorithms used: **Random Forest** (default), with comparison to others.
- Evaluation metrics: **R² Score**, **MAE**, **RMSE**

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/laptop-price-predictor.git

# Navigate into the directory
cd laptop-price-predictor

# Install the required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

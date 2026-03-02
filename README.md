# ✈️ Flight Price Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-DD0031?style=for-the-badge&logo=MachineLearning&logoColor=white)

An AI-powered flight price prediction web application that helps travelers estimate flight prices before making a booking decision. Built with Streamlit and Machine Learning.

## 🚀 Live Demo

**Access the live app here:** [SkyPrice - Flight Price Predictor](https://share.streamlit.io)

## 🌟 Features

- **🤖 ML-Based Predictions** - Advanced Gradient Boosting algorithm for accurate price estimates
- **📊 Interactive Analytics** - Beautiful charts showing price trends and distributions
- **💰 Price Insights** - Get recommendations on whether to book now or wait
- **🔥 Deal Finder** - Find the cheapest flights for your route
- **⚖️ Airline Comparison** - Compare prices across different airlines
- **📆 Advance Booking Planner** - Smart recommendations for when to book
- **🔔 Price Alerts** - Set alerts for routes you're interested in

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Python)
- **Backend:** Python
- **Machine Learning:** Scikit-learn (Gradient Boosting Regressor)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly

## 📁 Project Structure

```
Flight-Price-Predictor/
├── app.py                     # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── data/
│   └── business.csv          # Training data
├── model/
│   └── flight_price_model.pkl # Trained ML model
└── utils/
    └── preprocessing.py      # Data preprocessing utilities
```

## 🚀 Deployment to Streamlit Cloud

### Prerequisites

1. A GitHub account
2. A GitHub repository with your code

### Step-by-Step Deployment

#### 1. Prepare Your Repository

Make sure your repository has:
- `app.py` - Main application file
- `requirements.txt` - Dependencies
- `.streamlit/config.toml` - Configuration (optional but recommended)
- All required data and model files

#### 2. Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit - Flight Price Predictor"

# Add your GitHub repository
git remote add origin https://github.com/amirhamza15166/Flight-Price-Predictor.git

# Push to GitHub
git push -u origin main
```

#### 3. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `amirhamza15166/Flight-Price-Predictor`
5. Select branch: `main`
6. Main file path: `app.py`
7. Click "Deploy!"

Your app will be deployed and you'll get a URL like:
`https://flight-price-predictor-amirhamza15166.streamlit.app`

## 🔍 SEO Optimization

To help your app appear on Google search results:

### 1. Meta Tags (Add to app.py)

The app includes SEO-friendly title and description in the Streamlit configuration.

### 2. Content Optimization

The app includes rich content:
- Detailed feature descriptions
- Interactive analytics
- Informative About page
- Helpful tips for users

### 3. Sitemap Generation

For better indexing, create a `sitemap.txt`:

```
https://flight-price-predictor-amirhamza15166.streamlit.app
```

### 4. Google Search Console

1. Go to [search.google.com/search-console](https://search.google.com/search-console)
2. Add your Streamlit app URL as a property
3. Verify ownership through DNS or HTML file upload

## 📝 Usage

1. Select your travel date
2. Choose airline (Air India or Vistara)
3. Select departure and destination cities
4. Set departure/arrival times
5. Click "Predict Price" to get estimated flight cost

## 🔧 Development

### Local Setup

```bash
# Clone the repository
git clone https://github.com/amirhamza15166/Flight-Price-Predictor.git
cd Flight-Price-Predictor

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Model Details

- **Algorithm:** Gradient Boosting Regressor
- **Training Data:** Historical flight prices from Indian airlines
- **Features:** Airline, route, date, departure/arrival times, flight duration
- **Accuracy:** ~85% on validation data

## ⚠️ Disclaimer

Predictions are estimates based on historical data and should be used as a reference only. Actual prices may vary based on real-time market conditions, availability, and other factors. Always check with airlines or travel agencies for the most accurate pricing.

## 📄 License

MIT License

## 👤 Author

**Amir Hamza**
- GitHub: [@amirhamza15166](https://github.com/amirhamza15166)

## ⭐ Support

If you found this useful, please give this repo a ⭐️

---

Made with ❤️ using Streamlit

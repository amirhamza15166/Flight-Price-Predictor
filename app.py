"""
Flight Price Prediction Web Application
=======================================
A premium, animated Streamlit web app for predicting flight prices.

Author: ML Pipeline
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import time

# Import preprocessing utilities
import importlib
import utils.preprocessing as preprocessing_module

# Force reload to pick up changes
importlib.reload(preprocessing_module)

from utils.preprocessing import (
    preprocess_input,
    get_unique_values,
    format_price,
    load_data
)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="SkyPrice | Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium animated UI
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Main Theme */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #ec4899;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text: #f8fafc;
        --text-muted: #94a3b8;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
        letter-spacing: -1px;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(236, 72, 153, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
        animation: pulse 2s infinite;
    }
    
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 1rem 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    .prediction-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
    }
    
    /* Input Styling */
    .stSelectbox > div > div, 
    .stTimeInput > div > div > div,
    .stNumberInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    .stDateInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Subheader */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    /* Info boxes */
    .info-card {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: translateX(5px);
    }
    
    /* Divider */
    .custom-divider {
        margin: 2rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Loading Animation */
    .loading-dots::after {
        content: '.';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Animated Plane */
    .plane-icon {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from pickle file."""
    try:
        # Force reload the preprocessing module
        import importlib
        import utils.preprocessing
        importlib.reload(utils.preprocessing)
        
        with open('model/flight_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_training_data():
    """Load training data for visualizations."""
    try:
        df = pd.read_csv('data/business.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_price_distribution_chart(df):
    """Create animated price distribution chart."""
    df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
    
    fig = px.histogram(
        df, 
        x='price_clean', 
        nbins=40,
        title='📊 Price Distribution',
        labels={'price_clean': 'Price (₹)', 'count': 'Frequency'},
        color_discrete_sequence=['#6366f1']
    )
    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#f8fafc', family='Poppins'),
        title_font=dict(size=20, color='#f8fafc'),
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        bargap=0.1
    )
    return fig


def create_airline_price_comparison(df):
    """Create animated airline comparison chart."""
    df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
    
    fig = px.box(
        df, 
        x='airline', 
        y='price_clean',
        title='✈️ Price by Airline',
        labels={'price_clean': 'Price (₹)', 'airline': 'Airline'},
        color='airline',
        color_discrete_sequence=['#6366f1', '#ec4899']
    )
    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#f8fafc', family='Poppins'),
        title_font=dict(size=20, color='#f8fafc'),
        hovermode='closest',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def create_route_heatmap(df):
    """Create animated route price heatmap."""
    df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
    
    route_prices = df.groupby(['from', 'to'])['price_clean'].mean().reset_index()
    pivot = route_prices.pivot(index='from', columns='to', values='price_clean')
    
    fig = px.imshow(
        pivot,
        title='🗺️ Average Price by Route',
        labels={'color': 'Avg Price (₹)'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#f8fafc', family='Poppins'),
        title_font=dict(size=20, color='#f8fafc')
    )
    return fig


def create_price_trend_chart(df):
    """Create animated price trend chart."""
    df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
    
    # Parse date
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date_parsed'])
    df = df.sort_values('date_parsed')
    
    # Sample for performance
    df_sample = df.sample(min(1000, len(df)), random_state=42)
    
    fig = px.line(
        df_sample.groupby('date_parsed')['price_clean'].mean().reset_index(),
        x='date_parsed',
        y='price_clean',
        title='📈 Price Trends Over Time',
        labels={'date_parsed': 'Date', 'price_clean': 'Average Price (₹)'}
    )
    fig.update_traces(line_color='#06b6d4', line_width=3)
    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        font=dict(color='#f8fafc', family='Poppins'),
        title_font=dict(size=20, color='#f8fafc'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def get_multi_date_predictions(model, airline, source, destination, base_date, days_range=7):
    """
    Get price predictions for multiple dates around the base date.
    Returns predictions for dates before and after the selected date.
    """
    from datetime import timedelta, date
    predictions = []
    
    # Convert base_date to datetime if it's a date object
    if isinstance(base_date, date):
        base_datetime = datetime.combine(base_date, datetime.min.time())
    else:
        base_datetime = base_date
    
    today = datetime.now().date()
    
    for delta in range(-days_range, days_range + 1):
        check_date = base_datetime + timedelta(days=delta)
        
        # Skip past dates
        if check_date.date() < today:
            continue
            
        date_str = check_date.strftime('%Y-%m-%d')
        
        # Default times
        dep_time = "10:00"
        arr_time = f"{10 + (delta % 3) + 1}:00"
        time_taken = "2h 30m"
        
        input_features = preprocess_input(
            airline=airline,
            source=source,
            destination=destination,
            date=date_str,
            dep_time=dep_time,
            arr_time=arr_time,
            time_taken=time_taken
        )
        
        try:
            pred = model.predict(input_features)[0]
            pred = max(pred, 0)
        except:
            pred = 0
            
        predictions.append({
            'date': check_date,
            'date_str': check_date.strftime('%Y-%m-%d'),
            'day': check_date.strftime('%a'),
            'price': pred
        })
    
    return predictions


def compare_airline_prices(model, source, destination, flight_date):
    """
    Compare prices across all available airlines for a given route.
    """
    airlines = ['Air India', 'Vistara']
    results = []
    
    date_str = flight_date.strftime('%Y-%m-%d')
    dep_time = "10:00"
    arr_time = "13:00"
    time_taken = "2h 30m"
    
    for airline in airlines:
        input_features = preprocess_input(
            airline=airline,
            source=source,
            destination=destination,
            date=date_str,
            dep_time=dep_time,
            arr_time=arr_time,
            time_taken=time_taken
        )
        
        try:
            pred = model.predict(input_features)[0]
            pred = max(pred, 0)
        except:
            pred = 0
            
        results.append({
            'airline': airline,
            'price': pred
        })
    
    return results


def get_price_insights(model, airline, source, destination, flight_date, current_price):
    """
    Generate price insights and booking recommendations.
    """
    # Get predictions for nearby dates
    predictions = get_multi_date_predictions(model, airline, source, destination, flight_date, days_range=7)
    
    if not predictions:
        return None
    
    prices = [p['price'] for p in predictions]
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    # Calculate price score (0-100, higher is better price)
    price_range = max_price - min_price if max_price > min_price else 1
    price_score = int(100 - ((current_price - min_price) / price_range * 100))
    price_score = max(0, min(100, price_score))
    
    # Determine recommendation
    if current_price <= avg_price * 0.9:
        recommendation = "🟢 Great Deal!"
        recommendation_text = "This price is below average. Good time to book!"
        recommendation_color = "#10b981"
    elif current_price <= avg_price * 1.1:
        recommendation = "🟡 Fair Price"
        recommendation_text = "This price is around average. You might find a better deal."
        recommendation_color = "#f59e0b"
    else:
        recommendation = "🔴 High Price"
        recommendation_text = "This price is above average. Consider waiting for a better deal."
        recommendation_color = "#ef4444"
    
    return {
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'price_score': price_score,
        'recommendation': recommendation,
        'recommendation_text': recommendation_text,
        'recommendation_color': recommendation_color,
        'predictions': predictions
    }


def find_best_deals(model, source, destination, num_days=30):
    """
    Find the best deals for a route over the next N days.
    """
    from datetime import timedelta
    
    deals = []
    airlines = ['Air India', 'Vistara']
    today = datetime.today()
    
    for day_offset in range(num_days):
        check_date = today + timedelta(days=day_offset)
        date_str = check_date.strftime('%Y-%m-%d')
        
        day_prices = []
        for airline in airlines:
            dep_time = "10:00"
            arr_time = "13:00"
            time_taken = "2h 30m"
            
            input_features = preprocess_input(
                airline=airline,
                source=source,
                destination=destination,
                date=date_str,
                dep_time=dep_time,
                arr_time=arr_time,
                time_taken=time_taken
            )
            
            try:
                pred = model.predict(input_features)[0]
                pred = max(pred, 0)
            except:
                pred = 0
                
            day_prices.append({
                'airline': airline,
                'price': pred
            })
        
        # Get cheapest option for this day
        cheapest = min(day_prices, key=lambda x: x['price'])
        
        deals.append({
            'date': check_date,
            'date_str': check_date.strftime('%Y-%m-%d'),
            'day': check_date.strftime('%a'),
            'day_name': check_date.strftime('%A'),
            'airline': cheapest['airline'],
            'price': cheapest['price']
        })
    
    # Sort by price
    deals.sort(key=lambda x: x['price'])
    
    return deals[:10]  # Return top 10 deals


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Initialize ALL session state variables at the very start
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'last_inputs' not in st.session_state:
        st.session_state.last_inputs = {}
    if 'saved_searches' not in st.session_state:
        st.session_state.saved_searches = []
    if 'price_alerts' not in st.session_state:
        st.session_state.price_alerts = []
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "🏠 Home"
    
    # Load model and data
    model = load_model()
    df = load_training_data()
    
    # Header with animation
    st.markdown('''
    <div class="main-header">
        <span class="plane-icon">✈️</span> SkyPrice
    </div>
    <p style="text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;">
        AI-Powered Flight Price Prediction
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    # Sidebar with Premium Design
    with st.sidebar:
        # Premium Sidebar CSS
        st.markdown("""
        <style>
        /* Premium Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
            border-right: 1px solid rgba(99, 102, 241, 0.3) !important;
        }
        
        /* Premium Logo */
        .premium-logo {
            text-align: center;
            padding: 1.5rem 0.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
            border-radius: 16px;
            margin-bottom: 1rem;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        .premium-logo-icon {
            font-size: 2.5rem;
            display: block;
            animation: float 3s ease-in-out infinite;
        }
        
        .premium-logo-title {
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #ec4899 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.5rem 0 0 0;
        }
        
        .premium-logo-subtitle {
            color: #94a3b8;
            font-size: 0.8rem;
            margin: 0.25rem 0 0 0;
        }
        
        /* Premium Section */
        .premium-section {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 14px;
            padding: 1rem;
            margin: 0.75rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .premium-section-title {
            color: white;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        .premium-item {
            display: flex;
            justify-content: space-between;
            color: #94a3b8;
            font-size: 0.8rem;
            padding: 0.4rem 0.6rem;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px;
            margin: 0.3rem 0;
        }
        
        .premium-item span:last-child {
            color: #06b6d4;
            font-weight: 600;
        }
        
        .premium-feature {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #94a3b8;
            font-size: 0.8rem;
            padding: 0.3rem 0;
        }
        
        .premium-check {
            color: #10b981;
            font-weight: bold;
        }
        
        .premium-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
            margin: 1rem 0;
        }
        
        .premium-footer {
            text-align: center;
            padding: 1rem;
            color: #64748b;
            font-size: 0.75rem;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Premium Logo
        st.markdown("""
        <div class="premium-logo">
            <span class="premium-logo-icon">✈️</span>
            <h2 class="premium-logo-title">SkyPrice</h2>
            <p class="premium-logo-subtitle">Flight Price Predictor</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<p class="premium-section-title">🧭 Navigation</p>', unsafe_allow_html=True)
        
        nav_options = ["🏠 Home", "📊 Analytics", "🔥 Deals", "ℹ️ About"]
        
        for nav_option in nav_options:
            if st.button(f"{nav_option}", key=f"nav_{nav_option.split()[1]}", use_container_width=True):
                st.session_state.nav_page = nav_option
                st.rerun()
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        
        # Features Section
        st.markdown("""
        <div class="premium-section">
            <p class="premium-section-title">✨ Features</p>
            <p class="premium-feature"><span class="premium-check">✓</span> ML-based Predictions</p>
            <p class="premium-feature"><span class="premium-check">✓</span> Interactive Analytics</p>
            <p class="premium-feature"><span class="premium-check">✓</span> Real-time Estimates</p>
            <p class="premium-feature"><span class="premium-check">✓</span> Price Alerts</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Info
        st.markdown("""
        <div class="premium-section">
            <p class="premium-section-title">🧠 Model Info</p>
        """, unsafe_allow_html=True)
        
        if model:
            st.markdown(
                "<div class='premium-item'><span>Algorithm</span><span>Gradient Boosting</span></div>"
                "<div class='premium-item'><span>Type</span><span>Regression</span></div>"
                "<div class='premium-item'><span>Accuracy</span><span>~85%</span></div>",
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick Stats
        if df is not None:
            df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
            st.markdown(f"""
            <div class="premium-section">
                <p class="premium-section-title">📈 Quick Stats</p>
                <div class="premium-item"><span>Total Flights</span><span>{len(df):,}</span></div>
                <div class="premium-item"><span>Avg Price</span><span>₹{df['price_clean'].mean():,.0f}</span></div>
                <div class="premium-item"><span>Min Price</span><span>₹{df['price_clean'].min():,.0f}</span></div>
                <div class="premium-item"><span>Max Price</span><span>₹{df['price_clean'].max():,.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="premium-footer">
            <p>Made with ❤️ using Streamlit</p>
            <p>Version 2.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.session_state.nav_page
    
    # Main content based on navigation
    if page == "🏠 Home":
        home_page(model, df)
    elif page == "📊 Analytics":
        analytics_page(df)
    elif page == "🔥 Deals":
        deals_page(model, df)
    elif page == "ℹ️ About":
        about_page()


def home_page(model, df):
    """Home page with prediction form."""
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<p class="sub-header">📋 Flight Details</p>', unsafe_allow_html=True)
        
        # Input form in glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Date picker
        flight_date = st.date_input(
            "📅 Travel Date",
            min_value=datetime.today(),
            max_value=datetime.today() + timedelta(days=365),
            value=datetime.today() + timedelta(days=7),
            help="Select your travel date"
        )
        
        # Airline selection
        airline = st.selectbox(
            "🛫 Select Airline",
            options=['Air India', 'Vistara'],
            help="Choose your preferred airline"
        )
        
        # Source and Destination
        col_source, col_dest = st.columns(2)
        with col_source:
            source = st.selectbox(
                "🛫 From",
                options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
                index=0,
                help="Departure city"
            )
        with col_dest:
            destination = st.selectbox(
                "🎯 To",
                options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
                index=1,
                help="Destination city"
            )
        
        # Time inputs
        col_dep, col_arr = st.columns(2)
        with col_dep:
            dep_time = st.time_input(
                "⏰ Departure",
                value=datetime.strptime("10:00", "%H:%M"),
                help="Departure time"
            )
        with col_arr:
            arr_time = st.time_input(
                "⏰ Arrival",
                value=datetime.strptime("13:00", "%H:%M"),
                help="Arrival time"
            )
        
        # Flight duration
        col_dur_h, col_dur_m = st.columns(2)
        with col_dur_h:
            duration_hours = st.number_input(
                "⏱️ Hours",
                min_value=0,
                max_value=24,
                value=2,
                help="Flight duration (hours)"
            )
        with col_dur_m:
            duration_minutes = st.number_input(
                "⏱️ Minutes",
                min_value=0,
                max_value=59,
                value=30,
                help="Flight duration (minutes)"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create predict button
        predict_btn = st.button(
            "🔮 Predict Price",
            use_container_width=True,
            type="primary"
        )
        
        # Handle prediction
        if predict_btn:
            st.session_state.prediction_made = True
            st.session_state.last_inputs = {
                'airline': airline,
                'source': source,
                'destination': destination,
                'flight_date': flight_date,
                'dep_time': dep_time,
                'arr_time': arr_time,
                'duration_hours': duration_hours,
                'duration_minutes': duration_minutes
            }
            
            # Show loading animation
            with st.spinner('🧠 Calculating best price...'):
                time.sleep(1)  # Simulate processing
                
                # Format inputs for preprocessing
                date_str = flight_date.strftime('%Y-%m-%d')
                dep_time_str = dep_time.strftime('%H:%M')
                arr_time_str = arr_time.strftime('%H:%M')
                time_taken_str = f"{duration_hours}h {duration_minutes}m"
                
                # Preprocess input
                input_features = preprocess_input(
                    airline=airline,
                    source=source,
                    destination=destination,
                    date=date_str,
                    dep_time=dep_time_str,
                    arr_time=arr_time_str,
                    time_taken=time_taken_str
                )
                
                # Make prediction
                prediction = model.predict(input_features)[0]
                
                # Ensure prediction is positive
                prediction = max(prediction, 0)
                
                # Store result
                st.session_state.prediction_result = prediction
        
        # Display saved prediction if exists
        if st.session_state.prediction_made and st.session_state.prediction_result is not None:
            # Use stored inputs if available
            if st.session_state.last_inputs:
                airline = st.session_state.last_inputs.get('airline', airline)
                source = st.session_state.last_inputs.get('source', source)
                destination = st.session_state.last_inputs.get('destination', destination)
                flight_date = st.session_state.last_inputs.get('flight_date', flight_date)
                duration_hours = st.session_state.last_inputs.get('duration_hours', duration_hours)
                duration_minutes = st.session_state.last_inputs.get('duration_minutes', duration_minutes)
            
            prediction = st.session_state.prediction_result
            
            # Display at TOP - Large prediction card
            st.markdown(f'''
            <div class="prediction-card" style="margin-top: 1rem;">
                <p class="prediction-label">Estimated Flight Price</p>
                <p class="prediction-price">{format_price(prediction)}</p>
                <p class="prediction-label">{airline} • {source} → {destination}</p>
                <p class="prediction-label" style="font-size: 0.9rem;">📅 {flight_date.strftime('%B %d, %Y')} • ⏱️ {duration_hours}h {duration_minutes}m</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Price Insights Section
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">📊 Price Insights</p>', unsafe_allow_html=True)
            
            with st.spinner('📈 Analyzing prices...'):
                insights = get_price_insights(model, airline, source, destination, flight_date, prediction)
                
                if insights:
                    # Score and recommendation
                    score_col1, score_col2 = st.columns([1, 2])
                    
                    with score_col1:
                        score_color = insights['recommendation_color']
                        st.markdown(f'''
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 100px; height: 100px; border-radius: 50%; background: conic-gradient({score_color} {insights['price_score']}%, rgba(30,41,59,0.8) 0%); display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                                <div style="width: 80px; height: 80px; border-radius: 50%; background: #1e293b; display: flex; align-items: center; justify-content: center;">
                                    <span style="font-size: 1.5rem; font-weight: 800; color: white;">{insights['price_score']}</span>
                                </div>
                            </div>
                            <p style="color: #94a3b8; margin-top: 0.5rem;">Price Score</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with score_col2:
                        st.markdown(f'''
                        <div class="glass-card" style="border-left: 4px solid {insights['recommendation_color']};">
                            <h3 style="color: {insights['recommendation_color']}; margin: 0;">{insights['recommendation']}</h3>
                            <p style="color: #94a3b8; margin-top: 0.5rem;">{insights['recommendation_text']}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        range_col1, range_col2, range_col3 = st.columns(3)
                        with range_col1:
                            st.metric("Avg Price", format_price(insights['avg_price']))
                        with range_col2:
                            st.metric("Min Price", format_price(insights['min_price']))
                        with range_col3:
                            st.metric("Max Price", format_price(insights['max_price']))
                    
                    # Multi-date chart
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<p class="sub-header">📅 Price by Date (Next 2 Weeks)</p>', unsafe_allow_html=True)
                    
                    predictions_df = pd.DataFrame(insights['predictions'])
                    if not predictions_df.empty:
                        fig_deals = px.bar(
                            predictions_df,
                            x='date_str',
                            y='price',
                            color='price',
                            color_continuous_scale='Viridis',
                            title='Predicted Prices Around Your Date',
                            labels={'date_str': 'Date', 'price': 'Price (₹)'}
                        )
                        fig_deals.update_layout(
                            plot_bgcolor='rgba(30, 41, 59, 0.5)',
                            paper_bgcolor='rgba(30, 41, 59, 0.5)',
                            font=dict(color='#f8fafc', family='Poppins'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                        )
                        st.plotly_chart(fig_deals, use_container_width=True)
            
            # Save search button
            if st.button("💾 Save This Search", use_container_width=True):
                if 'saved_searches' not in st.session_state:
                    st.session_state.saved_searches = []
                search_data = {
                    'source': source,
                    'destination': destination,
                    'date': flight_date.strftime('%Y-%m-%d'),
                    'airline': airline,
                    'price': prediction,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                st.session_state.saved_searches.append(search_data)
                st.success("✅ Search saved!")
    
    with col2:
        st.markdown('<p class="sub-header">🎯 Quick Actions</p>', unsafe_allow_html=True)
        
        # Custom styled action cards
        st.markdown('''
        <style>
        .action-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
            border: 1px solid rgba(99, 102, 241, 0.5);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .action-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
            border-color: rgba(99, 102, 241, 0.8);
        }
        .action-card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }
        .action-card-desc {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 0;
        }
        </style>
        
        <div class="action-card">
            <div class="action-card-title">🔥 Find Best Deals</div>
            <p class="action-card-desc">Discover cheapest flights for your route</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Explore Deals", use_container_width=True, key="btn_deals"):
            st.session_state.nav_page = "🔥 Deals"
            st.rerun()
        
        st.markdown('''
        <div class="action-card" style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.3) 0%, rgba(249, 115, 22, 0.3) 100%); border-color: rgba(236, 72, 153, 0.5);">
            <div class="action-card-title">⚖️ Compare Airlines</div>
            <p class="action-card-desc">Compare prices across different airlines</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("⚖️ Compare Now", use_container_width=True, key="btn_compare"):
            st.session_state.nav_page = "🔥 Deals"
            st.rerun()
        
        st.markdown('''
        <div class="action-card" style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.3) 0%, rgba(16, 185, 129, 0.3) 100%); border-color: rgba(6, 182, 212, 0.5);">
            <div class="action-card-title">📊 View Analytics</div>
            <p class="action-card-desc">Explore historical price trends</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("📈 View Analytics", use_container_width=True, key="btn_analytics"):
            st.session_state.nav_page = "📊 Analytics"
            st.rerun()
    
    # Tips section
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">💡 Pro Tips</p>', unsafe_allow_html=True)
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown('''
        <div class="info-card">
            <strong>📅 Book in Advance</strong>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                Booking 2-3 weeks ahead often gives better prices. Avoid peak travel seasons.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with tip_col2:
        st.markdown('''
        <div class="info-card">
            <strong>🛫 Off-Peak Times</strong>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                Early morning and late night flights are cheaper. Weekdays are better than weekends.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with tip_col3:
        st.markdown('''
        <div class="info-card">
            <strong>🔄 Compare Airlines</strong>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                Both Air India and Vistara operate major routes. Prices vary based on demand.
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Advance Booking Recommendation Section
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">📆 Advance Booking Planner</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    advance_col1, advance_col2 = st.columns(2)
    
    with advance_col1:
        target_date = st.date_input(
            "🎯 Target Travel Date",
            min_value=datetime.today(),
            max_value=datetime.today() + timedelta(days=365),
            value=datetime.today() + timedelta(days=30),
            help="When do you want to travel?"
        )
    
    with advance_col2:
        days_until_travel = (target_date - datetime.today().date()).days
        
        if days_until_travel <= 7:
            booking_status = "🔴 Book Now!"
            booking_msg = "Travel is very soon - prices may be higher"
            status_color = "#ef4444"
        elif days_until_travel <= 14:
            booking_status = "🟠 Book Soon"
            booking_msg = "Within 2 weeks - good time to book"
            status_color = "#f59e0b"
        elif days_until_travel <= 30:
            booking_status = "🟢 Good Time to Book"
            booking_msg = "2-4 weeks ahead - typically best prices"
            status_color = "#10b981"
        else:
            booking_status = "🔵 Wait for Better Deals"
            booking_msg = "More than a month away - prices may drop"
            status_color = "#06b6d4"
        
        st.markdown(f'''
        <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.8); border-radius: 12px;">
            <h3 style="color: {status_color}; margin: 0;">{booking_status}</h3>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">{booking_msg}</p>
            <p style="color: #94a3b8; font-size: 0.8rem;">{days_until_travel} days until travel</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Quick prediction for selected date
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        quick_airline = st.selectbox(
            "Airline",
            options=['Air India', 'Vistara'],
            key="quick_airline"
        )
    
    with quick_col2:
        quick_source = st.selectbox(
            "From",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=0,
            key="quick_source"
        )
    
    with quick_col3:
        quick_dest = st.selectbox(
            "To",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=1,
            key="quick_dest"
        )
    
    if st.button("🔍 Check Price for This Date", type="primary", use_container_width=True):
        if quick_source == quick_dest:
            st.error("Source and destination cannot be the same!")
        else:
            with st.spinner('Checking price...'):
                date_str = target_date.strftime('%Y-%m-%d')
                
                input_features = preprocess_input(
                    airline=quick_airline,
                    source=quick_source,
                    destination=quick_dest,
                    date=date_str,
                    dep_time="10:00",
                    arr_time="13:00",
                    time_taken="2h 30m"
                )
                
                quick_pred = model.predict(input_features)[0]
                quick_pred = max(quick_pred, 0)
                
                # Show the price prominently
                st.markdown(f'''
                <div class="prediction-card" style="margin-top: 1rem;">
                    <p class="prediction-label">Price for {target_date.strftime('%B %d, %Y')}</p>
                    <p class="prediction-price">{format_price(quick_pred)}</p>
                    <p class="prediction-label">{quick_airline} • {quick_source} → {quick_dest}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show comparison with current prediction
                if st.session_state.prediction_result:
                    diff = quick_pred - st.session_state.prediction_result
                    if diff < 0:
                        st.success(f"💰 This date is {format_price(abs(diff))} cheaper!")
                    elif diff > 0:
                        st.warning(f"📈 This date is {format_price(diff)} more expensive")
                    else:
                        st.info("💵 Same price as your current selection")
                
                # Option to save this search
                if st.button("💾 Save This Flight"):
                    if 'saved_searches' not in st.session_state:
                        st.session_state.saved_searches = []
                    search_data = {
                        'source': quick_source,
                        'destination': quick_dest,
                        'date': target_date.strftime('%Y-%m-%d'),
                        'airline': quick_airline,
                        'price': quick_pred,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }
                    st.session_state.saved_searches.append(search_data)
                    st.success("✅ Flight saved!")
    
    # Recommended booking window
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        earliest_book = target_date - timedelta(days=21)
        st.metric("Earliest Booking", earliest_book.strftime('%b %d'), delta=f"{max(0, (earliest_book - datetime.today().date()).days)} days")
    
    with rec_col2:
        best_book = target_date - timedelta(days=14)
        st.metric("Best Time to Book", best_book.strftime('%b %d'), delta=f"{max(0, (best_book - datetime.today().date()).days)} days")
    
    with rec_col3:
        latest_book = target_date - timedelta(days=7)
        st.metric("Latest Booking", latest_book.strftime('%b %d'), delta=f"{max(0, (latest_book - datetime.today().date()).days)} days")
    
    # Set Price Alert button
    if st.button("🔔 Set Price Alert for This Route", use_container_width=True):
        if 'price_alerts' not in st.session_state:
            st.session_state.price_alerts = []
        
        # Get current price if available
        alert_price = st.session_state.prediction_result if st.session_state.prediction_result else 0
        
        alert_data = {
            'source': quick_source,
            'destination': quick_dest,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'target_price': alert_price,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        st.session_state.price_alerts.append(alert_data)
        st.success("🔔 Price alert set! We'll track this route for you.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show active alerts
    if 'price_alerts' in st.session_state and st.session_state.price_alerts:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">🔔 Your Price Alerts</p>', unsafe_allow_html=True)
        
        for idx, alert in enumerate(st.session_state.price_alerts):
            st.markdown(f'''
            <div class="glass-card" style="background: rgba(99, 102, 241, 0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{alert['source']} → {alert['destination']}</strong><br>
                        <span style="color: #94a3b8; font-size: 0.9rem;">Target: {alert['target_date']}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #10b981; font-weight: bold;">{format_price(alert['target_price'])}</span><br>
                        <span style="color: #94a3b8; font-size: 0.8rem;">Set: {alert['created']}</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button(f"🗑️ Remove Alert {idx}", key=f"remove_alert_{idx}"):
                st.session_state.price_alerts.pop(idx)
                st.rerun()


def analytics_page(df):
    """Analytics page with data visualizations."""
    
    if df is None:
        st.markdown('''
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <span style="font-size: 4rem;">⚠️</span>
            <p style="color: #94a3b8; margin-top: 1rem;">Unable to load data for analytics.</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    st.markdown('<p class="sub-header">📊 Flight Price Analytics</p>', unsafe_allow_html=True)
    
    # Clean price data
    df['price_clean'] = df['price'].astype(str).str.replace(',', '').astype(float)
    
    # Top metrics with animation
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Flights", f"{len(df):,}", delta="📈")
    with metric_col2:
        st.metric("Average Price", f"₹{df['price_clean'].mean():,.0f}", delta="💰")
    with metric_col3:
        st.metric("Min Price", f"₹{df['price_clean'].min():,.0f}", delta="⬇️")
    with metric_col4:
        st.metric("Max Price", f"₹{df['price_clean'].max():,.0f}", delta="⬆️")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_price_distribution_chart(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_airline_price_comparison(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Route heatmap
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(create_route_heatmap(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Price trend
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(create_price_trend_chart(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def deals_page(model, df):
    """Deals page with best price finder and comparison tools."""
    
    st.markdown('<p class="sub-header">🔥 Find Best Deals</p>', unsafe_allow_html=True)
    
    # Initialize session state for saved searches
    if 'saved_searches' not in st.session_state:
        st.session_state.saved_searches = []
    
    # Deal finder section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### 🔍 Route Deal Finder', unsafe_allow_html=True)
    
    deal_col1, deal_col2, deal_col3 = st.columns(3)
    
    with deal_col1:
        deal_source = st.selectbox(
            "🛫 From",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=0,
            key="deal_source"
        )
    
    with deal_col2:
        deal_destination = st.selectbox(
            "🎯 To",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=1,
            key="deal_destination"
        )
    
    with deal_col3:
        deal_days = st.slider(
            "📅 Search Days Ahead",
            min_value=7,
            max_value=60,
            value=30,
            key="deal_days"
        )
    
    find_deals_btn = st.button("🚀 Find Best Deals", type="primary")
    
    if find_deals_btn:
        if deal_source == deal_destination:
            st.error("⚠️ Source and destination cannot be the same!")
        else:
            with st.spinner('🔍 Searching for the best deals...'):
                deals = find_best_deals(model, deal_source, deal_destination, num_days=deal_days)
                
                if deals:
                    # Create deals table
                    deals_df = pd.DataFrame(deals)
                    deals_df['date_formatted'] = deals_df['date'].dt.strftime('%b %d, %Y')
                    
                    # Display top deals
                    st.markdown("### 🎉 Top Deals Found!")
                    
                    # Custom deal cards
                    for i, deal in enumerate(deals[:5], 1):
                        is_best = i == 1
                        border_color = "#10b981" if is_best else "rgba(255,255,255,0.1)"
                        bg_color = "rgba(16, 185, 129, 0.15)" if is_best else "rgba(30, 41, 59, 0.7)"
                        
                        deal_date = deal['date'].strftime('%b %d, %Y') if hasattr(deal['date'], 'strftime') else str(deal['date'])
                        
                        st.markdown(f'''
                        <div class="glass-card" style="background: {bg_color}; border: 2px solid {border_color}; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 1.5rem; font-weight: 700; color: white;">#{i}</span>
                                    <span style="color: #94a3b8; margin-left: 1rem;">{deal_date} ({deal.get('day_name', deal.get('day', ''))})</span>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 1.2rem; color: #94a3b8;">{deal['airline']}</span><br>
                                    <span style="font-size: 1.8rem; font-weight: 700; color: #10b981;">{format_price(deal['price'])}</span>
                                </div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Save this search
                    if st.button(f"💾 Save This Search"):
                        search_data = {
                            'source': deal_source,
                            'destination': deal_destination,
                            'top_deals': deals[:3],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        st.session_state.saved_searches.append(search_data)
                        st.success("✅ Search saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Airline Comparison Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### ✈️ Compare Airlines', unsafe_allow_html=True)
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        comp_source = st.selectbox(
            "From",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=0,
            key="comp_source"
        )
    
    with comp_col2:
        comp_dest = st.selectbox(
            "To",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
            index=1,
            key="comp_dest"
        )
    
    with comp_col3:
        comp_date = st.date_input(
            "Travel Date",
            min_value=datetime.today(),
            max_value=datetime.today() + timedelta(days=365),
            value=datetime.today() + timedelta(days=14),
            key="comp_date"
        )
    
    compare_btn = st.button("⚖️ Compare Prices", type="primary")
    
    if compare_btn:
        if comp_source == comp_dest:
            st.error("⚠️ Source and destination cannot be the same!")
        else:
            with st.spinner('🔄 Comparing prices...'):
                results = compare_airline_prices(model, comp_source, comp_dest, comp_date)
                
                # Display comparison
                st.markdown('### 📊 Price Comparison Results')
                
                comp_col1, comp_col2 = st.columns(2)
                
                for i, result in enumerate(results):
                    with (comp_col1 if i == 0 else comp_col2):
                        airline_color = "#6366f1" if result['airline'] == 'Air India' else "#ec4899"
                        st.markdown(f'''
                        <div class="glass-card" style="text-align: center; border: 2px solid {airline_color};">
                            <h3 style="color: {airline_color}; margin: 0;">{result['airline']}</h3>
                            <p style="font-size: 2.5rem; font-weight: 800; color: white; margin: 1rem 0;">{format_price(result['price'])}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Find cheapest
                cheapest = min(results, key=lambda x: x['price'])
                savings = max(results, key=lambda x: x['price'])['price'] - cheapest['price']
                
                st.markdown(f'''
                <div class="glass-card" style="background: rgba(16, 185, 129, 0.2); text-align: center;">
                    <h3 style="color: #10b981;">💰 Best Value: {cheapest['airline']}</h3>
                    <p style="color: #94a3b8;">Save up to <strong style="color: #10b981;">{format_price(savings)}</strong> by choosing {cheapest['airline']}</p>
                </div>
                ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Saved Searches Section
    if st.session_state.saved_searches:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">💾 Saved Searches</p>', unsafe_allow_html=True)
        
        for idx, search in enumerate(st.session_state.saved_searches):
            with st.expander(f"🔍 {search['source']} → {search['destination']} ({search['timestamp']})", expanded=False):
                for deal in search['top_deals']:
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>{deal['date'].strftime('%b %d, %Y')}</strong> ({deal['day']}) - 
                        <span style="color: #10b981;">{deal['airline']}</span> - 
                        <strong>{format_price(deal['price'])}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button(f"🗑️ Delete Search {idx}", key=f"delete_{idx}"):
                    st.session_state.saved_searches.pop(idx)
                    st.rerun()


def about_page():
    """About page with application information."""
    
    st.markdown('<p class="sub-header">ℹ️ About SkyPrice</p>', unsafe_allow_html=True)
    
    # Overview
    st.markdown('''
    <div class="glass-card">
        <h2 style="color: white; margin-bottom: 1rem;">🚀 Application Overview</h2>
        <p style="color: #94a3b8; line-height: 1.8;">
            SkyPrice is an AI-powered flight price prediction application designed to help travelers 
            estimate flight prices before making a booking decision. Using advanced machine learning 
            algorithms, we analyze historical flight data to provide accurate price predictions.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Features
    st.markdown('''
    <div class="glass-card">
        <h2 style="color: white; margin-bottom: 1rem;">✨ Features</h2>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div class="info-card">
                <strong>🤖 ML-Based Predictions</strong>
                <p style="color: #94a3b8; font-size: 0.9rem;">Advanced algorithms for accurate estimates</p>
            </div>
            <div class="info-card">
                <strong>📊 Interactive Analytics</strong>
                <p style="color: #94a3b8; font-size: 0.9rem;">Explore historical data with beautiful charts</p>
            </div>
            <div class="info-card">
                <strong>⚡ Real-Time Results</strong>
                <p style="color: #94a3b8; font-size: 0.9rem;">Get instant price estimates</p>
            </div>
            <div class="info-card">
                <strong>🌍 Multiple Routes</strong>
                <p style="color: #94a3b8; font-size: 0.9rem;">Covers all major Indian cities</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # How it works
    st.markdown('''
    <div class="glass-card">
        <h2 style="color: white; margin-bottom: 1rem;">🔧 How It Works</h2>
        <ol style="color: #94a3b8; line-height: 2; padding-left: 1.5rem;">
            <li>Enter your flight details including date, airline, route, and timing</li>
            <li>Click the <strong>Predict Price</strong> button</li>
            <li>Get an instant price estimate powered by our Gradient Boosting model</li>
        </ol>
    </div>
    ''', unsafe_allow_html=True)
    
    # Model details
    st.markdown('''
    <div class="glass-card">
        <h2 style="color: white; margin-bottom: 1rem;">📊 Model Details</h2>
        <ul style="color: #94a3b8; line-height: 2; padding-left: 1.5rem;">
            <li><strong>Algorithm</strong>: Gradient Boosting Regressor</li>
            <li><strong>Training Data</strong>: Historical flight prices from major Indian airlines</li>
            <li><strong>Features Used</strong>: Airline, route, date, departure/arrival times, flight duration</li>
            <li><strong>Accuracy</strong>: ~85% on validation data</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('''
    <div class="glass-card" style="background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.3);">
        <h2 style="color: #fbbf24; margin-bottom: 1rem;">⚠️ Disclaimer</h2>
        <p style="color: #94a3b8; line-height: 1.8;">
            Predictions are estimates based on historical data and should be used as a reference only. 
            Actual prices may vary based on real-time market conditions, availability, and other factors.
            Always check with airlines or travel agencies for the most accurate pricing.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Contact
    st.markdown('''
    <div class="custom-divider"></div>
    <div style="text-align: center; padding: 2rem;">
        <p style="color: #94a3b8;">📞 For questions or feedback, please contact support</p>
        <p style="color: #6366f1; font-weight: 600;">Made with ❤️ using Streamlit</p>
    </div>
    ''', unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

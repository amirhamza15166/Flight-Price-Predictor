"""
Preprocessing utilities for the flight price prediction model.
This module handles data transformation and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os


# Label encodings based on training data
AIRLINE_ENCODING = {
    'Air India': 0,
    'Vistara': 1
}

CITY_ENCODING = {
    'Delhi': 0,
    'Mumbai': 1,
    'Bangalore': 2,
    'Kolkata': 3,
    'Hyderabad': 4,
    'Chennai': 5
}

# Reverse mappings for decoding
AIRLINE_DECODING = {v: k for k, v in AIRLINE_ENCODING.items()}
CITY_DECODING = {v: k for k, v in CITY_ENCODING.items()}


def load_data(file_path='data/business.csv'):
    """Load the flight data for reference."""
    df = pd.read_csv(file_path)
    return df


def get_airline_num_code(airline):
    """Get the numeric code for an airline (legacy function)."""
    airline_codes = {
        'Air India': 868,
        'Vistara': 624
    }
    return airline_codes.get(airline, 868)


def get_airline_encoding(airline):
    """Get the label encoding for an airline."""
    return AIRLINE_ENCODING.get(airline, 0)


def get_city_encoding(city):
    """Get the label encoding for a city."""
    return CITY_ENCODING.get(city, 0)


def parse_time_to_minutes(time_str):
    """Convert time string (e.g., '02h 15m') to minutes."""
    if pd.isna(time_str):
        return 0
    time_str = str(time_str).strip()
    try:
        if 'h' in time_str:
            parts = time_str.split('h')
            hours = int(parts[0].strip()) if parts[0].strip() else 0
            minutes = int(parts[1].replace('m', '').strip()) if len(parts) > 1 and parts[1].strip() else 0
        else:
            hours = 0
            minutes = int(time_str.replace('m', '').strip())
        return hours * 60 + minutes
    except:
        return 0


def parse_time(time_str):
    """Parse time string to extract hour and minute."""
    if pd.isna(time_str):
        return 0, 0
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            return hour, minute
        return 0, 0
    except:
        return 0, 0


def encode_date(date_str):
    """Encode date to ordinal value."""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.toordinal()
    except:
        return 738000  # Default to a reasonable date


def preprocess_input(
    airline,
    source,
    destination,
    date,
    dep_time,
    arr_time,
    time_taken
):
    """
    Transform user input into model features.
    
    Parameters:
    -----------
    airline : str
        Airline name (Air India, Vistara)
    source : str
        Source city
    destination : str
        Destination city
    date : str
        Date in format YYYY-MM-DD
    dep_time : str
        Departure time in format HH:MM
    arr_time : str
        Arrival time in format HH:MM
    time_taken : str
        Time taken in format 'XXh XXm'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with features in correct order for model prediction
    """
    # Parse date
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        month = date_obj.month
        year = date_obj.year
        date_ordinal = date_obj.toordinal()
    except:
        month = 1
        year = 2024
        date_ordinal = 738000
    
    # Get airline encoding (label encoding)
    airline_encoded = get_airline_encoding(airline)
    
    # Get numeric code (legacy feature)
    num_code = get_airline_num_code(airline)
    
    # Get city encodings
    source_encoded = get_city_encoding(source)
    dest_encoded = get_city_encoding(destination)
    
    # Parse departure time
    dep_hr, dep_min = parse_time(dep_time)
    
    # Parse arrival time
    arr_hr, arr_min = parse_time(arr_time)
    
    # Parse time taken to minutes
    time_taken_mins = parse_time_to_minutes(time_taken)
    
    # Create feature array in the exact order the model expects
    # Model expects: ['airline' 'num_code' 'from' 'to' 'date' 'month' 'year' 'arr_hour' 'arr_min' 'dep_hr' 'dep_min' 'time_taken_mins']
    features = pd.DataFrame([[
        airline_encoded,      # airline (encoded)
        num_code,             # num_code
        source_encoded,       # from (encoded)
        dest_encoded,         # to (encoded)
        date_ordinal,         # date (ordinal)
        month,                # month
        year,                 # year
        arr_hr,               # arr_hour
        arr_min,              # arr_min
        dep_hr,               # dep_hr
        dep_min,              # dep_min
        time_taken_mins       # time_taken_mins
    ]], columns=[
        'airline', 'num_code', 'from', 'to', 'date', 'month', 'year',
        'arr_hour', 'arr_min', 'dep_hr', 'dep_min', 'time_taken_mins'
    ])
    
    return features


def get_unique_values(df):
    """Extract unique values for dropdown menus."""
    return {
        'airlines': df['airline'].unique().tolist(),
        'cities': sorted(df['from'].unique().tolist()),
        'airline_codes': {airline: get_airline_num_code(airline) for airline in df['airline'].unique()}
    }


def format_price(price):
    """Format price with Indian Rupee symbol and comma separator."""
    try:
        price = float(price)
        return f"₹{price:,.0f}"
    except:
        return f"₹0"

import numpy as np
import pandas as pd # Optional: if using pandas Series/DataFrame internally

def calculate_sma(prices: list[float] | pd.Series, period: int) -> float | None:
    if not isinstance(prices, (list, pd.Series)):
        raise TypeError("Prices must be a list or pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
        
    if len(prices) < period:
        return None
    
    if isinstance(prices, list):
        return np.mean(prices[-period:])
    elif isinstance(prices, pd.Series):
        return prices.iloc[-period:].mean()
    return None # Should not be reached given type checks

def calculate_rsi(prices: list[float] | pd.Series, period: int = 14) -> float | None:
    if not isinstance(prices, (list, pd.Series)):
        raise TypeError("Prices must be a list or pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")

    if len(prices) < period + 1: # Need at least period + 1 prices to calculate deltas
        return None

    if isinstance(prices, list):
        price_series = pd.Series(prices) # Convert list to Series for easier calculation
    else:
        price_series = prices

    delta = price_series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    if loss.iloc[-1] == 0: # Avoid division by zero; if loss is 0, RSI is 100
        return 100.0
    
    rs = gain.iloc[-1] / loss.iloc[-1]
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # The first few RSI values will be NaN due to rolling window.
    # We are interested in the last calculated RSI.
    if pd.isna(rsi): # Check if the result is NaN (e.g. not enough data points yet for full rolling mean)
        return None 
        
    return rsi

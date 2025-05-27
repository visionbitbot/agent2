import pytest
import pandas as pd
from chimera.feature_engineering.alpha_factors.technical_indicators import calculate_sma, calculate_rsi

class TestCalculateSMA:
    def test_sma_basic_list(self):
        assert calculate_sma([1, 2, 3, 4, 5], 3) == 4.0
        assert calculate_sma([1, 2, 3, 4, 5, 6], 3) == 5.0

    def test_sma_basic_series(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert calculate_sma(s, 3) == 4.0

    def test_sma_not_enough_data(self):
        assert calculate_sma([1, 2], 3) is None

    def test_sma_empty_list(self):
        assert calculate_sma([], 3) is None

    def test_sma_invalid_period(self):
        with pytest.raises(ValueError):
            calculate_sma([1, 2, 3, 4, 5], 0)
        with pytest.raises(ValueError):
            calculate_sma([1, 2, 3, 4, 5], -1)

    def test_sma_invalid_prices_type(self):
        with pytest.raises(TypeError):
            calculate_sma("not a list", 3) #type: ignore

class TestCalculateRSI:
    # Test data from: https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    # (Using their example calculations for validation)
    prices_example = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64
    ] # 20 data points

    def test_rsi_basic_list(self):
        # For period 14, we need 15 data points for the first RSI.
        # The example usually shows the smoothed RSI after the initial period.
        # The current implementation calculates RSI using simple rolling mean of gains/losses.
        # This might differ slightly from some platform's smoothed RSI initially.
        # Let's test with enough data.
        rsi_val = calculate_rsi(self.prices_example, 14) 
        assert rsi_val is not None
        # Example value from a calculator using simple Wilder's smoothing (which pandas rolling can approximate)
        # For the given data and period 14, the RSI is around 53-58 depending on exact smoothing.
        # A simple rolling mean RSI might be different. Let's check for a reasonable range.
        # For the last point of prices_example (45.64), a 14-period RSI is approx 53.0-55.0
        # Using the provided code's logic (which is a valid way to compute non-smoothed RSI):
        # Last 15 points: [46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
        # RSI for this sequence with period 14 using the code's logic
        # Note: The rolling mean needs 'period' points. Delta needs 'period' points.
        # So, the first RSI value is at index 'period'.
        # Let's use a known sequence for easier verification
        known_prices = [45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 45.15, 45.00, 45.30] # 15 points
        # RSI for this set of 15 points, period 14:
        # delta: [0.24, -0.19, 0.14, -0.42, 0.67, 0.0, -0.28, 0.03, 0.38, -0.19, -0.58, -0.49, -0.15, 0.3]
        # gain_avg = sum of gains / 14
        # loss_avg = sum of abs(losses) / 14
        # gains: 0.24, 0.14, 0.67, 0.03, 0.38, 0.3 => sum = 1.76
        # losses: 0.19, 0.42, 0.28, 0.19, 0.58, 0.49, 0.15 => sum = 2.3
        # avg_gain = 1.76/14 = 0.1257
        # avg_loss = 2.3/14 = 0.1643
        # rs = 0.1257 / 0.1643 = 0.765
        # rsi = 100 - (100 / (1 + 0.765)) = 100 - 56.65 = 43.35
        assert calculate_rsi(known_prices, 14) == pytest.approx(43.35, abs=0.1)


    def test_rsi_basic_series(self):
        s = pd.Series(self.prices_example)
        rsi_val = calculate_rsi(s, 14)
        assert rsi_val is not None
        # Should be same as list version if data is identical
        known_prices_s = pd.Series([45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 45.15, 45.00, 45.30])
        assert calculate_rsi(known_prices_s, 14) == pytest.approx(43.35, abs=0.1)


    def test_rsi_not_enough_data(self):
        assert calculate_rsi([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 14) is None # Needs 15 for period 14

    def test_rsi_all_prices_same(self):
        # If all prices are same, gain and loss are 0. RS is undefined or 1 (depends on handling 0 loss).
        # If loss is 0, RSI should be 100.
        assert calculate_rsi([10] * 20, 14) == 100.0

    def test_rsi_all_prices_increasing(self):
        # All gains, no losses. RSI should be 100.
        assert calculate_rsi(list(range(10, 30)), 14) == 100.0

    def test_rsi_all_prices_decreasing(self):
        # All losses, no gains. RSI should be 0.
        # avg_gain = 0, avg_loss > 0 => RS = 0 => RSI = 0
        rsi_val = calculate_rsi(list(range(30, 10, -1)), 14)
        assert rsi_val is not None
        assert rsi_val == pytest.approx(0.0, abs=0.01)


    def test_rsi_invalid_period(self):
        with pytest.raises(ValueError):
            calculate_rsi(self.prices_example, 0)
        with pytest.raises(ValueError):
            calculate_rsi(self.prices_example, -1)
            
    def test_rsi_invalid_prices_type(self):
        with pytest.raises(TypeError):
            calculate_rsi("not a list or series", 14) #type: ignore

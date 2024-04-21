from flask import Flask, request, jsonify, render_template
import yfinance as yf
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from candlestick import candlestick

app = Flask(__name__, static_url_path='/static')


# Suppress warnings
warnings.filterwarnings("ignore")

@app.route('/')
def index():
    return render_template('cost.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    cmp = data.get('ticker') + '.NS'  # Add '.NS' for NSE stocks
    recent_data = yf.download(cmp, period="2y", interval="1d")
    recent_data.drop('Adj Close', axis=1, inplace=True)

    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        # Calculate short-term and long-term EMAs
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()  
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

        # Calculate MACD line
        macd_line = short_ema - long_ema

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line

        return macd_line, signal_line, macd_histogram

    def calculate_rsi(data, window=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stochastic_oscillator(data, window=14):
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        stochastic_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
        stochastic_d = stochastic_k.rolling(window=3).mean()  # 3-day smoothing for %D line
        return stochastic_k, stochastic_d

    def calculate_moving_averages(data, short_window=50, long_window=200):
        # Calculate short-term and long-term moving averages
        short_ma = data['Close'].rolling(window=short_window).mean()
        long_ma = data['Close'].rolling(window=long_window).mean()
        return short_ma, long_ma

    def calculate_volume_indicators(data, window=20):
        # Calculate volume moving average
        volume_ma = data['Volume'].rolling(window=window).mean()
        # Calculate volume rate of change
        volume_roc = data['Volume'].pct_change(window)
        return volume_ma, volume_roc

    def calculate_volatility_indicators(data, window=20):
        # Calculate Bollinger Bands
        std_dev = data['Close'].rolling(window=window).std()
        upper_band = data['Close'].rolling(window=window).mean() + 2 * std_dev
        lower_band = data['Close'].rolling(window=window).mean() - 2 * std_dev
        # Calculate Average True Range (ATR)
        high_low_range = data['High'] - data['Low']
        true_range = np.maximum(np.maximum((data['High'] - data['Low']).abs(), (data['High'] - data['Close'].shift()).abs()), (data['Low'] - data['Close'].shift()).abs())
        atr = true_range.rolling(window=window).mean()
        b = (data['Close']-lower_band)/(upper_band-lower_band)
        return upper_band, lower_band, atr , b

    def williams_percent_r(data, period=14):
        high = data['High']
        low = data['Low']
        close = data['Close']

        # Calculate %R
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        percent_r = -100 * (highest_high - close) / (highest_high - lowest_low)

        return percent_r

    # Calculate technical indicators
    macd_line, signal_line, macd_histogram = calculate_macd(recent_data)
    rsi = calculate_rsi(recent_data)
    stochastic_k, stochastic_d = calculate_stochastic_oscillator(recent_data)
    short_ma, long_ma = calculate_moving_averages(recent_data)
    volume_ma, volume_roc = calculate_volume_indicators(recent_data)
    upper_band, lower_band, atr, b = calculate_volatility_indicators(recent_data)

    # Add calculated indicators to the DataFrame
    recent_data['MACD_Line'] = macd_line
    recent_data['Signal_Line'] = signal_line
    recent_data['MACD_Histogram'] = macd_histogram
    recent_data['RSI'] = rsi
    recent_data['%K'] = stochastic_k
    recent_data['%D'] = stochastic_d
    recent_data['Short_MA'] = short_ma
    recent_data['Long_MA'] = long_ma
    recent_data['Volume_MA'] = volume_ma
    recent_data['Volume_ROC'] = volume_roc
    recent_data['Upper_Band'] = upper_band
    recent_data['Lower_Band'] = lower_band
    recent_data['ATR'] = atr
    recent_data['%B'] = b
    recent_data['%R']=williams_percent_r(recent_data)

    recent_data= recent_data.dropna(subset=['Long_MA'])
    data = recent_data.copy()
    data.reset_index(inplace=True)

    data['Date'] = pd.to_datetime(data['Date'])
    # data['Datetime'] = pd.to_datetime(data['Datetime'])

    # candles_df = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    candles_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    candles_df.columns = ['T', 'open', 'high', 'low', 'close', 'volume']

    # Defining the list of candlestick patterns you want to detect
    patterns = {
        "inverted_hammer": 'InvertedHammer',
        'doji_star':'DojiStar',
        'bearish_harami':'BearishHarami',
        'bullish_harami':'BullishHarami',
        'dark_cloud_cover':'DarkCloudCover',
        'doji':'Doji',
        'dragonfly_doji':'DragonflyDoji',
        'bearish_engulfing':'BearishEngulfing',
        'bullish_engulfing':'BullishEngulfing',
        'hammer':'Hammer',
        'morning_star':'MorningStar',
        'morning_star_doji':'MorningStarDoji',
        'piercing_pattern':'PiercingPattern',
        'rain_drop':'RainDrop',
        'rain_drop_doji':'RainDropDoji',
        'star':'Star',
        'shooting_star':'ShootingStar',
        'hanging_man':"HangingMan",
        'gravestone_doji':"GravestoneDoji"
    }

    for pattern, target in patterns.items():
        # Apply the candlestick pattern
        candles_df = getattr(candlestick, pattern)(candles_df, target=target)

    detected_patterns = []

    for index, row in candles_df.iterrows():
        pattern_detected = "None"
        for pattern, detected in row[6:].items():
            if detected:
                pattern_detected = pattern
                break
        detected_patterns.append(pattern_detected)

    candles_df['pattern'] = detected_patterns

    for pattern in patterns.values():
        candles_df.drop(pattern, axis=1, inplace=True)

    # Create a dictionary mapping dates to candlestick patterns from candles_df
    pattern_dict = dict(zip(candles_df['T'], candles_df['pattern']))

    # Add a new column 'pattern' to recent_data and fill it with NaN values initially
    recent_data['pattern'] = "None"

    for index, row in recent_data.iterrows():
        date_index = row.name  # Get the index of the row
        # Check if the date index exists in pattern_dict
        if date_index in pattern_dict:
            recent_data.at[date_index, 'pattern'] = pattern_dict[date_index]


    recent_data['Price_Move_1'] = np.where(recent_data['Close'] > recent_data['Open'], 1, -1)

    recent_data['Next_5_Days_Avg_Diff'] = recent_data['Close'].rolling(window=5).mean().shift(-5) - recent_data['Close']

    recent_data.dropna(subset=['Next_5_Days_Avg_Diff'], inplace=True)

    recent_data['Price_Move'] = np.where(recent_data['Next_5_Days_Avg_Diff'] > 0, 1, -1)

    # One-hot encode the "pattern" column
    # X = pd.get_dummies(recent_data[features], columns=['pattern'])
    unqValues = recent_data["pattern"].unique()
    for i in range(len(unqValues)):
        recent_data["pattern"].replace(unqValues[i], i, inplace= True)


    from sklearn.ensemble import RandomForestClassifier
    # Create an instance of RandomForestClassifier

    features = ['MACD_Histogram', 'RSI', '%K', '%D', '%R',"pattern","%B","Signal_Line","Volume_ROC","Volume",'Open', 'High', 'Low', 'Close'] 

    X = recent_data[features]
    y = recent_data['Price_Move']
    y[y == -1] = 0
    rf_classifier = RandomForestClassifier()

    # Train the classifier
    rf_classifier.fit(X[:-1], y[1:])

    # Predict the movement for the next trading day
    y_next = rf_classifier.predict(X.iloc[-1].values.reshape(1, -1))

        # Return prediction
    if y_next == 0:
        return jsonify({"prediction": "Downward"})
    elif y_next == 1:
        return jsonify({"prediction": "Upward"})
    

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)

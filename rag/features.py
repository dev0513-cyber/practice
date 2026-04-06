

def add_features(df):
    df = df.copy()

    df["return_1d"] = df["price"].pct_change(1)
    df["return_3d"] = df["price"].pct_change(3)
    df["return_7d"] = df["price"].pct_change(7)

    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_14"] = df["price"].rolling(14).mean()
    df["ma_30"] = df["price"].rolling(30).mean()

    df["std_7"] = df["price"].rolling(7).std()
    df["std_14"] = df["price"].rolling(14).std()

    df["price_to_ma7"] = df["price"] / df["ma_7"]
    df["price_to_ma14"] = df["price"] / df["ma_14"]

    df["volume_ma_7"] = df["volume"].rolling(7).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_7"]

    # RSI
    delta = df["price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["target_price"] = df["price"].shift(-1)
    df["target_direction"] = (df["target_price"] > df["price"]).astype(int)

    df = df.dropna()
    return df


FEATURE_COLS = [
    "return_1d", "return_3d", "return_7d",
    "ma_7", "ma_14", "ma_30",
    "std_7", "std_14",
    "price_to_ma7", "price_to_ma14",
    "volume_ratio", "rsi_14",
]

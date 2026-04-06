
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

from features import FEATURE_COLS

MODEL_DIR = "models"
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
DIR_MODEL_PATH = os.path.join(MODEL_DIR, "direction_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def train_models(df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Use all but last row (last row has no target since we shifted)
    train_df = df.iloc[:-1]

    X = train_df[FEATURE_COLS].values
    y_price = train_df["target_price"].values
    y_dir = train_df["target_direction"].values

    split = int(len(X) * 0.8)
    X_train_raw, X_val_raw = X[:split], X[split:]
    yp_train, yp_val = y_price[:split], y_price[split:]
    yd_train, yd_val = y_dir[:split], y_dir[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    price_model = RandomForestRegressor(n_estimators=100, random_state=42)
    price_model.fit(X_train, yp_train)

    dir_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dir_model.fit(X_train, yd_train)

    mae = mean_absolute_error(yp_val, price_model.predict(X_val))
    acc = accuracy_score(yd_val, dir_model.predict(X_val))
    print(f"Validation MAE (price): ${mae:,.0f}")
    print(f"Validation Accuracy (direction): {acc:.2%}")

    joblib.dump(price_model, PRICE_MODEL_PATH)
    joblib.dump(dir_model, DIR_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return price_model, dir_model, scaler


def load_models():
    price_model = joblib.load(PRICE_MODEL_PATH)
    dir_model = joblib.load(DIR_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return price_model, dir_model, scaler


def models_exist():
    return all(os.path.exists(p) for p in [PRICE_MODEL_PATH, DIR_MODEL_PATH, SCALER_PATH])


def predict(df, price_model, dir_model, scaler, direction_threshold=0.02):
    latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest)

    predicted_price = price_model.predict(latest_scaled)[0]
    current_price = df["price"].iloc[-1]
    price_change_pct = (predicted_price - current_price) / current_price

    direction_prob = dir_model.predict_proba(latest_scaled)[0][1]

    if price_change_pct > direction_threshold:
        signal = "UP"
    elif price_change_pct < -direction_threshold:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "price_change_pct": price_change_pct * 100,
        "direction_prob_up": direction_prob * 100,
        "signal": signal,
    }

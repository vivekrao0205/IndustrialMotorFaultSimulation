# Motor Aging Prediction (Software-Based)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis

# PARAMETERS

FS = 1000
T = 1
t = np.linspace(0, T, FS)

BASE_FREQ = 50
np.random.seed(42)

# AGING SIGNAL GENERATION

def motor_signal(age_level):
    """
    age_level: 0 (New) → 1 (Fully aged)
    """
    base_signal = np.sin(2 * np.pi * BASE_FREQ * t)

    noise = age_level * 0.5 * np.random.randn(len(t))
    vibration = age_level * 0.3 * np.sin(2 * np.pi * 120 * t)
    load_fluctuation = 1 + age_level * 0.2 * np.sin(2 * np.pi * 3 * t)

    return base_signal * load_fluctuation + vibration + noise

# AGING STAGES

new_motor = motor_signal(0.1)
mid_aged_motor = motor_signal(0.5)
old_motor = motor_signal(0.9)

# TIME DOMAIN COMPARISON

plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(new_motor)
plt.title("New Motor")
plt.grid()

plt.subplot(3,1,2)
plt.plot(mid_aged_motor)
plt.title("Mid-Aged Motor")
plt.grid()

plt.subplot(3,1,3)
plt.plot(old_motor)
plt.title("Old / Aging Motor")
plt.grid()

plt.suptitle("Motor Aging – Time Domain Comparison")
plt.tight_layout()
plt.show()

# FEATURE EXTRACTION

def extract_features(signal):
    return {
        "RMS": np.sqrt(np.mean(signal**2)),
        "Variance": np.var(signal),
        "Peak": np.max(np.abs(signal)),
        "Skewness": skew(signal),
        "Kurtosis": kurtosis(signal)
    }

features = {
    "New": extract_features(new_motor),
    "Mid-Aged": extract_features(mid_aged_motor),
    "Old": extract_features(old_motor)
}

df = pd.DataFrame(features)
print("\nAGING FEATURE TABLE:\n")
print(df)

# FEATURE TREND VISUALIZATION

df.T.plot(kind="bar", figsize=(12,5))
plt.title("Feature Trend with Motor Aging")
plt.ylabel("Feature Value")
plt.grid()
plt.show()

# AGING PREDICTION (RULE-BASED)

def predict_motor_age(signal):
    f = extract_features(signal)

    if f["Variance"] < 0.15 and f["Kurtosis"] < 3:
        return "New Motor"
    elif f["Variance"] < 0.35:
        return "Mid-Aged Motor"
    else:
        return "Old / Critical Aging Motor"

print("\nAGING PREDICTION RESULTS:")
print("New Motor       →", predict_motor_age(new_motor))
print("Mid-Aged Motor  →", predict_motor_age(mid_aged_motor))
print("Old Motor       →", predict_motor_age(old_motor))

# REAL-TIME AGING MONITORING

WINDOW = 250

print("\nREAL-TIME MOTOR AGING MONITORING:")
for i in range(0, len(old_motor) - WINDOW, WINDOW):
    segment = old_motor[i:i+WINDOW]
    print(f"Window {i}-{i+WINDOW} → {predict_motor_age(segment)}")

# AGING INDEX CALCULATION

def aging_index(signal):
    f = extract_features(signal)
    return (
        0.4 * f["RMS"] +
        0.4 * f["Variance"] +
        0.2 * abs(f["Kurtosis"])
    )

# Simulate gradual aging
age_levels = np.linspace(0.1, 1.0, 10)
aging_values = []

for age in age_levels:
    sig = motor_signal(age)
    aging_values.append(aging_index(sig))

# AGING INDEX vs TIME GRAPH (WITH THRESHOLDS)

plt.figure(figsize=(10,5))
plt.plot(age_levels, aging_values, marker="o", label="Aging Index")

plt.axhline(y=0.8, color="green", linestyle="--", label="Safe Zone")
plt.axhline(y=1.4, color="orange", linestyle="--", label="Warning Zone")
plt.axhline(y=2.0, color="red", linestyle="--", label="Critical Zone")

plt.xlabel("Motor Age Level (New → Old)")
plt.ylabel("Aging Index")
plt.title("Motor Aging Index Trend Over Time")
plt.legend()
plt.grid()
plt.show()

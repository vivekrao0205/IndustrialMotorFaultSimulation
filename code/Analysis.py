# Motor Fault Detection & Comparison (Software-Based)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# PARAMETERS
FS = 1000                 # Sampling frequency
T = 1                     # Duration (seconds)
t = np.linspace(0, T, FS)

BASE_FREQ = 50            # Supply frequency

# SIGNAL GENERATION
healthy = np.sin(2 * np.pi * BASE_FREQ * t)

bearing = healthy + 0.4 * np.sin(2 * np.pi * 300 * t)

rotor = healthy * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))

stator = healthy + 0.3 * np.random.randn(len(t))

# TIME DOMAIN COMPARISON
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.plot(healthy)
plt.title("Healthy Motor")
plt.grid()

plt.subplot(2,2,2)
plt.plot(bearing)
plt.title("Bearing Fault")
plt.grid()

plt.subplot(2,2,3)
plt.plot(rotor)
plt.title("Rotor Fault")
plt.grid()

plt.subplot(2,2,4)
plt.plot(stator)
plt.title("Stator Fault")
plt.grid()

plt.suptitle("Time Domain Comparison of Motor Conditions")
plt.tight_layout()
plt.show()

# FFT FUNCTION

def plot_fft(signal, title):
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N//2]
    freqs = np.fft.fftfreq(N, 1/FS)[:N//2]
    plt.plot(freqs, fft_vals)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()


# FREQUENCY DOMAIN COMPARISON

plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plot_fft(healthy, "Healthy FFT")

plt.subplot(2,2,2)
plot_fft(bearing, "Bearing Fault FFT")

plt.subplot(2,2,3)
plot_fft(rotor, "Rotor Fault FFT")

plt.subplot(2,2,4)
plot_fft(stator, "Stator Fault FFT")

plt.tight_layout()
plt.show()

# FEATURE EXTRACTION

def extract_features(signal):
    return {
        "RMS": np.sqrt(np.mean(signal**2)),
        "Mean": np.mean(signal),
        "Variance": np.var(signal),
        "Peak": np.max(np.abs(signal)),
        "Skewness": skew(signal),
        "Kurtosis": kurtosis(signal)
    }

features = {
    "Healthy": extract_features(healthy),
    "Bearing": extract_features(bearing),
    "Rotor": extract_features(rotor),
    "Stator": extract_features(stator)
}

df = pd.DataFrame(features)
print("\nFEATURE COMPARISON TABLE:\n")
print(df)


# FEATURE BAR GRAPH

df.T.plot(kind="bar", figsize=(12,5))
plt.title("Feature Comparison of Motor Conditions")
plt.ylabel("Feature Value")
plt.grid()
plt.show()


# RULE-BASED CLASSIFIER

def classify(signal):
    f = extract_features(signal)

    if f["Variance"] > 0.25 and f["Peak"] > 1.5:
        return "Bearing Fault"
    elif f["RMS"] > 1.1:
        return "Rotor Fault"
    elif f["Variance"] > 0.15:
        return "Stator Fault"
    else:
        return "Healthy Motor"

print("\nCLASSIFICATION RESULTS:")
print("Healthy  →", classify(healthy))
print("Bearing  →", classify(bearing))
print("Rotor    →", classify(rotor))
print("Stator   →", classify(stator))


# REAL-TIME SLIDING WINDOW

WINDOW = 250

print("\nREAL-TIME SIMULATION OUTPUT:")
for i in range(0, len(bearing) - WINDOW, WINDOW):
    segment = bearing[i:i+WINDOW]
    print(f"Window {i}-{i+WINDOW} → {classify(segment)}")

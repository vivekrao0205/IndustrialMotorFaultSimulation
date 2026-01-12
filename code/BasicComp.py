import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fs = 100
t = np.linspace(0, 1, fs)

healthy = np.sin(2 * np.pi * 50 * t)
bearing = healthy + 0.4 * np.sin(2 * np.pi * 300 * t)
rotor   = healthy * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
stator  = healthy + 0.3 * np.random.randn(len(t))

plt.figure(figsize=(10,4))
plt.plot(healthy, label="Healthy")
plt.plot(bearing, label="Bearing Fault")
plt.legend()
plt.title("Motor Current Signals")
plt.show()

def extract_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    mean = np.mean(signal)
    var = np.var(signal)
    peak = np.max(np.abs(signal))
    return rms, mean, var, peak

features = {
    "Healthy": extract_features(healthy),
    "Bearing": extract_features(bearing),
    "Rotor": extract_features(rotor),
    "Stator": extract_features(stator)
}

df = pd.DataFrame(features, index=["RMS", "Mean", "Variance", "Peak"])
print(df)

def classify(signal):
    rms, mean, var, peak = extract_features(signal)

    if var > 0.25 and peak > 1.5:
        return "Bearing Fault"
    elif rms > 1.2 and mean > 0.1:
        return "Rotor Fault"
    elif var > 0.15:
        return "Stator Fault"
    else:
        return "Healthy Motor"

print("Healthy:", classify(healthy))
print("Bearing:", classify(bearing))
print("Rotor:", classify(rotor))
print("Stator:", classify(stator))

window = 250
print("\nReal-Time Simulation Output:")
for i in range(0, len(healthy), window):
    segment = healthy[i:i+window]
    print(classify(segment))

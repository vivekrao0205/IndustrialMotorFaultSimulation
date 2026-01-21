import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.fft import fft
from scipy.stats import kurtosis, skew, entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===================== GLOBAL PARAMETERS =====================

FS = 5000
DURATION = 1.0
TIME = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
BASE_FREQ = 50
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===================== SIGNAL GENERATOR =====================

class MotorSignalGenerator:
    def __init__(self, t, base_freq):
        self.t = t
        self.base_freq = base_freq

    def healthy(self):
        return np.sin(2 * np.pi * self.base_freq * self.t)

    def bearing_fault(self, severity):
        return (self.healthy()
                + severity * np.sin(2 * np.pi * 300 * self.t)
                + 0.05 * np.random.randn(len(self.t)))

    def rotor_fault(self, severity):
        return self.healthy() * (1 + severity * np.sin(2 * np.pi * 5 * self.t))

    def stator_fault(self, severity):
        return self.healthy() + severity * np.random.randn(len(self.t))

    def apply_load_variation(self, signal):
        return signal * (1 + 0.2 * np.sin(2 * np.pi * 2 * self.t))

# ===================== FEATURE EXTRACTION =====================

class FeatureExtractor:
    def __init__(self, fs):
        self.fs = fs

    def extract(self, signal):
        signal = signal / np.max(np.abs(signal))  # normalization
        N = len(signal)

        window = np.hanning(N)
        fft_vals = np.abs(fft(signal * window))[:N // 2]
        freqs = np.fft.fftfreq(N, 1 / self.fs)[:N // 2]

        psd = fft_vals**2 / np.sum(fft_vals**2)

        return [
            np.sqrt(np.mean(signal**2)),                 # RMS
            np.mean(signal),
            np.var(signal),
            np.max(np.abs(signal)),                      # Peak
            np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),  # Crest factor
            skew(signal),
            kurtosis(signal),
            freqs[np.argmax(fft_vals)],                  # Dominant frequency
            np.sum(fft_vals**2),                          # Spectral energy
            entropy(psd),                                 # Spectral entropy
            np.sum(fft_vals[freqs < 100]**2),             # Low band energy
            np.sum(fft_vals[(freqs >= 100) & (freqs < 500)]**2),
            np.sum(fft_vals[freqs >= 500]**2)
        ]

    @staticmethod
    def feature_names():
        return [
            "RMS", "Mean", "Variance", "Peak", "CrestFactor",
            "Skewness", "Kurtosis", "DominantFreq",
            "SpectralEnergy", "SpectralEntropy",
            "BandLow", "BandMid", "BandHigh"
        ]

# ===================== DATASET BUILDER =====================

def build_dataset(generator, extractor, samples=40):
    X, y = [], []

    for _ in range(samples):
        signals = {
            "Healthy": generator.healthy(),
            "Bearing": generator.bearing_fault(np.random.uniform(0.3, 0.7)),
            "Rotor": generator.rotor_fault(np.random.uniform(0.2, 0.5)),
            "Stator": generator.stator_fault(np.random.uniform(0.3, 0.6))
        }

        for label, sig in signals.items():
            sig = generator.apply_load_variation(sig)
            X.append(extractor.extract(sig))
            y.append(label)

    return np.array(X), np.array(y)

# ===================== FFT PLOT =====================

def plot_fft(signal, title):
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N // 2]
    freqs = np.fft.fftfreq(N, 1 / FS)[:N // 2]
    plt.plot(freqs, fft_vals)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

# ===================== MAIN =====================

if __name__ == "__main__":

    generator = MotorSignalGenerator(TIME, BASE_FREQ)
    extractor = FeatureExtractor(FS)

    signals = {
        "Healthy": generator.healthy(),
        "Bearing": generator.bearing_fault(0.5),
        "Rotor": generator.rotor_fault(0.4),
        "Stator": generator.stator_fault(0.5)
    }

    # -------- Time Domain --------
    plt.figure(figsize=(12, 6))
    for k, v in signals.items():
        plt.plot(v[:1000], label=k)
    plt.title("Time Domain Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # -------- FFT Domain --------
    plt.figure(figsize=(12, 6))
    for i, (k, v) in enumerate(signals.items(), 1):
        plt.subplot(2, 2, i)
        plot_fft(v, f"{k} FFT")
    plt.tight_layout()
    plt.show()

    # -------- Feature Comparison --------
    feat_data = {k: extractor.extract(v) for k, v in signals.items()}
    df_feat = pd.DataFrame(feat_data, index=extractor.feature_names())
    df_feat.T[["RMS", "Variance", "Peak", "BandHigh"]].plot(kind="bar", figsize=(12, 5))
    plt.title("Feature Comparison")
    plt.grid()
    plt.show()

    # -------- ML--------
    X, y = build_dataset(generator, extractor)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

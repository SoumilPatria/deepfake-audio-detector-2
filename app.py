import os
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import streamlit as st

# ----------------------------------------------------------------------
# Config  (must match notebook.ipynb)
# ----------------------------------------------------------------------
SR     = 16000
N_MELS = 128
WIN    = 128
SILENCE_RMS = 0.005   # clips quieter than this are treated as "no usable speech"
MODEL_PATH     = "deepfake_audio_model_v3.pth"
THRESHOLD_PATH = "threshold.txt"
MODEL_URL     = "https://huggingface.co/SoumilPatria/deepfake_audio_detection/resolve/main/deepfake_audio_model_v3.pth"
THRESHOLD_URL = "https://huggingface.co/SoumilPatria/deepfake_audio_detection/resolve/main/threshold.txt"

# ----------------------------------------------------------------------
# Model architecture  (identical to training)
# ----------------------------------------------------------------------
class DeepfakeAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1);   self.bn1 = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for features  (copied verbatim from notebook.ipynb)
# ----------------------------------------------------------------------
def wav_to_mel(audio, sr=SR, n_mels=N_MELS):
    m = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return m.astype(np.float32)

def mel_windows(mel, win=WIN):
    if mel.shape[1] < win:
        mel = np.pad(mel, ((0, 0), (0, win - mel.shape[1])), mode='constant')
        return mel[None, :, :]
    n = mel.shape[1] // win
    return np.stack([mel[:, i * win:(i + 1) * win] for i in range(n)])

# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _download(url, path):
    r = requests.get(url, allow_redirects=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        _download(MODEL_URL, MODEL_PATH)
    model = DeepfakeAudioCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_threshold():
    try:
        if not os.path.exists(THRESHOLD_PATH):
            _download(THRESHOLD_URL, THRESHOLD_PATH)
        return float(open(THRESHOLD_PATH).read().strip())
    except Exception:
        return 0.5  # fallback if no calibrated threshold is hosted

# ----------------------------------------------------------------------
# Inference  (per-file sliding window: flag fake if ANY segment looks synthetic)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, audio, threshold):
    wins = mel_windows(wav_to_mel(audio))
    x = torch.tensor(wins, dtype=torch.float32).unsqueeze(1)
    fake = F.softmax(model(x), dim=1)[:, 1].numpy()
    fake_prob = float(fake.max())
    return {
        "fake_prob": fake_prob,
        "real_prob": 1.0 - fake_prob,
        "is_fake": fake_prob >= threshold,
        "n_windows": int(len(fake)),
        "per_window_fake": fake,
    }

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def fmt_prob(p):
    """Show probabilities with enough precision that a non-zero value never
    displays as a flat 0.0% (which made the verdict look self-contradictory)."""
    pct = p * 100.0
    if pct == 0:
        return "0%"
    if pct >= 0.1:
        return f"{pct:.1f}%"
    if pct >= 1e-4:
        return f"{pct:.4f}%"
    return f"{pct:.1e}%"

st.set_page_config(page_title="Deepfake Detector", page_icon="🎙️")
st.title("🎙️ Deepfake Audio Detector")
st.caption("Classifies a speech recording as Genuine (Human) or Deepfake (AI-Generated).")

model = load_model()
threshold = load_threshold()

uploaded_file = st.file_uploader("Upload audio", type=["wav", "flac", "mp3", "m4a", "ogg"])
if uploaded_file:
    temp_filename = "temp_audio"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        st.audio(uploaded_file)
        audio, _ = librosa.load(temp_filename, sr=SR)
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        if audio.size < SR // 2:  # < 0.5 s of usable audio
            st.warning("⚠️ Clip too short to analyse reliably (need at least ~0.5 s).")
        elif rms < SILENCE_RMS:   # silence / near-silent -> can't classify
            st.warning(
                "⚠️ Inconclusive: this clip is silent or has no usable speech, "
                "so it can't be classified. Please upload a clip with audible speech."
            )
        else:
            res = predict(model, audio, threshold)
            st.write("### Results")
            col1, col2 = st.columns(2)
            col1.metric("Genuine (Human)", fmt_prob(res["real_prob"]))
            col2.metric("Deepfake (AI)",   fmt_prob(res["fake_prob"]))

            if res["is_fake"]:
                st.error(f"🚨 Result: Deepfake Detected  (deepfake score {fmt_prob(res['fake_prob'])} ≥ threshold)")
            else:
                st.success(f"✅ Result: Genuine Audio  (deepfake score {fmt_prob(res['fake_prob'])} < threshold)")

            st.caption(
                f"Analysed {res['n_windows']} window(s) across the whole clip. "
                f"Score = highest deepfake probability over all windows. "
                f"Decision threshold = {threshold:.2e}."
            )
            if res["n_windows"] > 1:
                with st.expander("Per-window deepfake probability"):
                    st.bar_chart(res["per_window_fake"])
    except Exception as e:
        st.error(f"Prediction Error: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

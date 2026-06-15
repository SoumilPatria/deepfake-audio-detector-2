# 🎙️ Deepfake Audio Detection (MARS Open Project 2026)

## 📖 Overview
This repository contains an end-to-end machine learning pipeline and an interactive web application designed to classify speech recordings as either **Genuine (Human)** or **Deepfake (AI-Generated)**. Built as a deliverable for the MARS Open Projects 2026, this system provides a reliable tool for identifying synthetic voice patterns in real-time.

## 🚀 Live App
**Access the live web application here:** [https://deepfake-audio-detector.streamlit.app/](https://deepfake-audio-detector-mgubhu2gegq9g5eb7mw8zy.streamlit.app/)

---

## 🔬 Methodology & Architecture
The project follows a deep learning pipeline for audio classification:

1. **Dataset & Splits:** Trained on the `for-norm` (loudness-normalised, full-length) subset of the Fake-or-Real Dataset, using the dataset's own speaker-disjoint `training / validation / testing` folders so the reported metrics reflect true held-out generalization (not memorization of a single folder).
2. **Feature Extraction:** Raw waveforms (resampled to 16 kHz) are converted to **normalised log-Mel spectrograms** (`librosa`, 128 mel bands, min-max scaled). A single shared feature function (`wav_to_mel` + `mel_windows`) is used by both training and the app so they can never drift.
3. **Windowing:** Each clip is split into non-overlapping 128-frame (~4 s) windows. Training samples a random window per file (with **SpecAugment** + light noise); inference scores **every** window and flags the clip as fake if any window looks synthetic.
4. **Model Architecture:** A custom **Convolutional Neural Network (CNN)** in PyTorch (3 conv+BatchNorm blocks → FC head) over the spectrogram windows.
5. **Regularization:** Weight decay + dropout, with **early stopping on validation EER**; the decision threshold is calibrated at the validation EER operating point (saved to `threshold.txt`).



---

## 📊 Performance Metrics
Reported on the **held-out `for-norm/testing` split** (run the evaluation cell in `notebook.ipynb` to regenerate these after training). Fill in your numbers from that cell:

| Metric | Achieved Score | Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | _TBD_ | >= 80% | — |
| **Equal Error Rate (EER)** | _TBD_ | <= 12% | — |
| **Macro F1 Score** | _TBD_ | >= 80% | — |
| **Genuine Accuracy** | _TBD_ | >= 75% | — |
| **Deepfake Accuracy** | _TBD_ | >= 75% | — |

> Note: earlier reported numbers (~97% accuracy) came from a random split of a single training folder and overstated real-world performance. The pipeline now evaluates on the dataset's dedicated, speaker-disjoint testing split.

The notebook saves `training_history.png` and `confusion_matrix.png` — add them here after a run.


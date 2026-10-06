# DeepGuard — AI Deepfake Detector

DeepGuard is a lightweight web application that analyzes **images** and **audio** files and estimates whether they are authentic or AI-generated. It combines a Flask backend with pretrained Hugging Face models and a clean, responsive dark-themed interface.

> **Disclaimer:** No deepfake detector is 100% accurate. DeepGuard provides a probabilistic estimate and should be used as a decision-support tool, not as definitive proof.

---

## Features

- **Image analysis** — detects AI-generated or manipulated images using an ensemble of pretrained models for more stable results.
- **Audio analysis** — detects synthetic or cloned voices using a Wav2Vec2-based classifier.
- **Clear results** — verdict, fake/real probability, confidence score, and per-model details.
- **Modern UI** — drag-and-drop upload, live preview, image/audio tabs, and an animated confidence meter.
- **Secure uploads** — file type whitelist, size limit, randomized temporary filenames, and automatic cleanup.
- **Resilient backend** — models load lazily on first use; a model that fails to load is skipped instead of crashing the app.

---

## Tech Stack

| Layer      | Technology                                   |
|------------|----------------------------------------------|
| Backend    | Python, Flask, Flask-CORS                    |
| ML         | Hugging Face Transformers, PyTorch           |
| Audio      | Librosa, SoundFile                           |
| Imaging    | Pillow, NumPy                                |
| Frontend   | HTML5, CSS3, Vanilla JavaScript              |

### Models Used

| Task  | Model                                                                                                            |
|-------|------------------------------------------------------------------------------------------------------------------|
| Image | [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection)    |
| Image | [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector)                                        |
| Audio | [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2)  |

For images, the final fake probability is the average of all successfully loaded models.

---

## Project Structure

```
deepfake_detector/
├── app.py                # Flask backend and model inference
├── requirements.txt      # Python dependencies
├── static/
│   └── style.css         # Stylesheet
├── templates/
│   └── index.html        # Frontend (UI + JavaScript)
└── uploads/              # Temporary upload folder (auto-created)
```

---

## Getting Started

### Prerequisites

- **Python 3.10 – 3.12** (recommended; PyTorch support on newer versions may lag)
- An internet connection on first run (models are downloaded once and cached)
- *(Optional)* [FFmpeg](https://ffmpeg.org/) for `.m4a` / `.mp4` audio files

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install PyTorch (CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install the remaining dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

> The first scan downloads the models (several hundred MB in total) and may take a while. Subsequent scans are much faster.

---

## Usage

1. Choose the **Image** or **Audio** tab.
2. Drag and drop a file, or click to browse.
3. Click **Scan Now**.
4. Review the verdict, fake/real probabilities, confidence, and model details.

**Supported formats**

- Images: `JPG`, `JPEG`, `PNG`, `WEBP`, `BMP`
- Audio: `WAV`, `MP3`, `M4A`, `OGG`, `FLAC`, `MP4`

**Limits:** maximum file size is 25 MB; audio analysis uses the first 30 seconds.

---

## API Reference

All endpoints accept `multipart/form-data` with a single `file` field.

| Method | Endpoint             | Description                                 |
|--------|----------------------|---------------------------------------------|
| GET    | `/`                  | Web interface                               |
| GET    | `/api/health`        | Health check                                |
| POST   | `/api/analyze`       | Analyze an image                            |
| POST   | `/api/analyze-audio` | Analyze an audio file                       |
| POST   | `/detect`, `/analyze`| Auto-detect file type and analyze           |

### Example

```bash
curl -X POST -F "file=@photo.jpg" http://127.0.0.1:5000/api/analyze
```

### Example Response

```json
{
  "verdict": "Authentic / Real",
  "is_fake": false,
  "confidence": 91,
  "fake_probability": 9.0,
  "real_probability": 91.0,
  "details": "deepfake_vs_real_image_detection: 8% fake | sdxl-detector: 10% fake",
  "model": "DeepGuard Image Ensemble (2 models)"
}
```

---

## Configuration

Key settings are defined at the top of `app.py`:

| Setting             | Default | Description                                        |
|---------------------|---------|----------------------------------------------------|
| `IMAGE_MODELS`      | 2 models| Hugging Face models used for image ensemble        |
| `AUDIO_MODELS`      | 1 model | Hugging Face models used for audio                 |
| `FAKE_THRESHOLD`    | `0.5`   | Probability above which a file is marked as fake   |
| `MAX_FILE_MB`       | `25`    | Maximum upload size                                |
| `MAX_AUDIO_SECONDS` | `30`    | Audio length analyzed                              |

If real files are flagged as fake too often, try raising `FAKE_THRESHOLD` (e.g. `0.6`) or swapping in a different model.

---

## Security Notes

- Uploaded filenames are never used on disk; audio files are saved with random UUID names and deleted immediately after analysis.
- Strict file extension whitelist and request size limit.
- Debug mode is disabled by default.
- Results are rendered with `textContent` in the frontend to avoid HTML injection.

---

## Limitations

- Detection accuracy depends on the underlying pretrained models and may vary across generators, compression levels, and recording conditions.
- Heavily compressed, edited, or low-resolution media can reduce accuracy.
- Newer generative models may not be recognized by older detectors.
- Results are estimates and may include false positives or false negatives.

---

## Roadmap

- Video deepfake detection
- Batch file analysis
- Model fine-tuning on custom datasets
- Docker support
- Result history and export

---

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, and open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

**Abubakr** — Computer Science student specializing in cybersecurity.

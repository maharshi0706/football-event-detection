# frontend/api_client.py
import requests
from typing import Optional

API_URL = "http://localhost:8000"
TIMEOUT = 60


def get_samples() -> list[dict]:
    """Fetch list of available sample clips from API."""
    try:
        res = requests.get(f"{API_URL}/samples", timeout=5)
        if res.ok:
            return res.json()["clips"]
        return []
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API. Is it running on port 8000?")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch samples: {e}")


def get_sample_video(filename: str) -> Optional[bytes]:
    """Fetch raw video bytes for a sample clip preview."""
    try:
        res = requests.get(f"{API_URL}/sample-video/{filename}", timeout=5)
        return res.content if res.ok else None
    except Exception:
        return None


def predict_upload(video_bytes: bytes, filename: str = "clip.mp4") -> list[dict]:
    """
    Send uploaded video bytes to API for inference.
    Returns list of {class, confidence} dicts.
    """
    try:
        res = requests.post(
            f"{API_URL}/predict",
            files={"video": (filename, video_bytes, "video/mp4")},
            timeout=TIMEOUT
        )
        if res.ok:
            return res.json()["predictions"]
        error = res.json().get("detail", "Unknown API error")
        raise RuntimeError(f"API error: {error}")

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API. Run: uvicorn api.main:app --reload")
    except requests.exceptions.Timeout:
        raise TimeoutError("Request timed out — clip may be too long.")


def predict_sample(filename: str) -> list[dict]:
    """
    Request inference on a pre-uploaded sample clip by filename.
    Returns list of {class, confidence} dicts.
    """
    try:
        res = requests.post(
            f"{API_URL}/predict-sample",
            json={"filename": filename},
            timeout=TIMEOUT
        )
        if res.ok:
            return res.json()["predictions"]
        error = res.json().get("detail", "Unknown API error")
        raise RuntimeError(f"API error: {error}")

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API. Run: uvicorn api.main:app --reload")
    except requests.exceptions.Timeout:
        raise TimeoutError("Request timed out.")
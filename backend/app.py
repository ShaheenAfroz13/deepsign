from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import os
import sqlite3
import json
import tempfile

from identity_matcher import match_identity, get_users_with_embeddings, cosine_similarity_pct, RECOGNITION_MODEL, SIMILARITY_THRESHOLD, DB_PATH, UPLOADS_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

IMG_SIZE = 299
MAX_SEQ_LENGTH = 15
NUM_FEATURES = 2048

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MODEL_REGISTRY = {
    "deepfakes": {
        "label": "Deepfakes",
        "description": "Identity replacement via autoencoder",
        "model_path": "models/deepfakes.keras",
    },
    "face2face": {
        "label": "Face2Face",
        "description": "Facial reenactment transfer",
        "model_path": "models/face2face.keras",
    },
    "faceswap": {
        "label": "FaceSwap",
        "description": "Geometry-based face swap",
        "model_path": "models/faceswap.keras",
    },
    "faceshifter": {
        "label": "FaceShifter",
        "description": "High-fidelity identity-preserving swap",
        "model_path": "models/faceshifter.keras",
    },
    "neuraltextures": {
        "label": "NeuralTextures",
        "description": "Texture-based facial rendering",
        "model_path": "models/neuraltextures.keras",
    },
}

_model_cache: dict = {}

# Model architecture helpers

def build_architecture():
    from tensorflow import keras
    base = keras.applications.Xception(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    preprocess = keras.applications.xception.preprocess_input
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    out = base(preprocess(inp))
    feature_extractor = keras.Model(inp, out, name="feature_extractor")

    frame_inp = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_inp = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    x = keras.layers.GRU(64, return_sequences=True)(frame_inp, mask=mask_inp)
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model([frame_inp, mask_inp], x)
    return model, feature_extractor


def get_model(model_key: str):
    if model_key in _model_cache:
        return _model_cache[model_key]

    from tensorflow import keras
    info = MODEL_REGISTRY[model_key]
    model_path = info.get("model_path")

    if model_path and os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"[OK] {model_key}: loaded {model_path}")
    else:
        model, _ = build_architecture()
        print(f"[WARN] {model_key}: no model found — placeholder mode")

    base = keras.applications.Xception(
        weights="imagenet", include_top=False,
        pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    preprocess = keras.applications.xception.preprocess_input
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_extractor = keras.Model(inp, base(preprocess(inp)), name="feature_extractor")

    _model_cache[model_key] = (model, feature_extractor)
    return model, feature_extractor


# Video processing helpers 


def crop_face(frame, expand_ratio=0.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    ew, eh = int(w * expand_ratio), int(h * expand_ratio)
    x1, y1 = max(x - ew, 0), max(y - eh, 0)
    x2, y2 = min(x + w + ew, frame.shape[1]), min(y + h + eh, frame.shape[0])
    return frame[y1:y2, x1:x2]


def load_video_frames(path, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = crop_face(frame)
        if face is None:
            continue
        face_rgb = cv2.resize(face, (IMG_SIZE, IMG_SIZE))[:, :, ::-1]
        frames.append(face_rgb)
        if len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames) if frames else None


def predict_single_model(model_key: str, frames: np.ndarray):
    model, feature_extractor = get_model(model_key)
    info = MODEL_REGISTRY[model_key]

    length = min(MAX_SEQ_LENGTH, len(frames))
    features = feature_extractor(frames[:length], training=False).numpy()

    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features[0, :length] = features
    frame_mask[0, :length] = True

    pred = float(model.predict([frame_features, frame_mask], verbose=0)[0][0])
    label = "FAKE" if pred > 0.5 else "REAL"
    confidence = pred if label == "FAKE" else 1 - pred

    has_weights = (
        info.get("model_path") is not None
        and os.path.exists(info.get("model_path", ""))
    )

    return {
        "key": model_key,
        "label": info["label"],
        "description": info["description"],
        "verdict": label,
        "fake_probability": round(pred * 100, 1),
        "real_probability": round((1 - pred) * 100, 1),
        "confidence": round(confidence * 100, 1),
        "frames_analyzed": length,
        "has_weights": has_weights,
    }



# Routes



@app.route("/debug/match", methods=["POST"])
def debug_match():
    """
    POST a single image file (field name: "image") to see raw similarity scores
    against every registered user. Use this to tune SIMILARITY_THRESHOLD.
    curl -X POST http://localhost:5000/debug/match -F "image=@face.jpg"
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400

    file = request.files["image"]
    ext  = os.path.splitext(file.filename)[1].lower() or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        probe_path = tmp.name

    try:
        from deepface import DeepFace
    except ImportError:
        return jsonify({"error": "deepface not installed — run: pip install deepface tf-keras"}), 500

    with_embeddings, without_embeddings = get_users_with_embeddings()
    results = []

    try:
        probe_result = DeepFace.represent(
            img_path=probe_path,
            model_name=RECOGNITION_MODEL,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )
        if not probe_result:
            return jsonify({"error": "No face detected in uploaded image"}), 422

        probe_vec = probe_result[0]["embedding"]

        for user in with_embeddings:
            sim = cosine_similarity_pct(probe_vec, user["embedding_vector"])
            results.append({
                "user_id":  user["id"],
                "username": user["username"],
                "similarity": sim,
                "would_match": sim >= SIMILARITY_THRESHOLD,
                "method": "stored_embedding",
            })

        for user in without_embeddings:
            gallery_path = os.path.join(UPLOADS_DIR, user["face"])
            if not os.path.exists(gallery_path):
                continue
            try:
                r = DeepFace.verify(
                    img1_path=probe_path, img2_path=gallery_path,
                    model_name=RECOGNITION_MODEL, detector_backend="opencv",
                    enforce_detection=False, align=True, distance_metric="cosine",
                )
                sim = round(max(0.0, 1.0 - r.get("distance", 1.0)) * 100, 1)
                results.append({
                    "user_id":  user["id"],
                    "username": user["username"],
                    "similarity": sim,
                    "would_match": sim >= SIMILARITY_THRESHOLD,
                    "method": "live_verify",
                })
            except Exception as e:
                results.append({"username": user["username"], "error": str(e)})

    finally:
        if os.path.exists(probe_path):
            os.unlink(probe_path)

    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return jsonify({
        "model": RECOGNITION_MODEL,
        "threshold": SIMILARITY_THRESHOLD,
        "results": results,
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(allowed)}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Frame extraction
        frames = load_video_frames(tmp_path)
        if frames is None or len(frames) == 0:
            return jsonify({"error": "No faces detected in video"}), 422

        # ── Deepfake detection 
        results = []
        errors = []
        for key in MODEL_REGISTRY:
            try:
                result = predict_single_model(key, frames)
                results.append(result)
            except Exception as e:
                errors.append({"key": key, "error": str(e)})

        fake_votes = sum(1 for r in results if r["verdict"] == "FAKE")
        real_votes = len(results) - fake_votes
        avg_fake_prob = round(
            sum(r["fake_probability"] for r in results) / len(results), 1
        ) if results else 0.0
        avg_real_prob = round(100 - avg_fake_prob, 1)
        overall_verdict = "FAKE" if fake_votes > real_votes else "REAL"

        #  Identity matching, only runs when FAKE is detected 
        # identity_matcher owns its own frame extraction pipeline independently
        identity_match = None
        if overall_verdict == "FAKE":
            try:
                identity_match = match_identity(tmp_path)
            except Exception as e:
                identity_match = {"error": f"Identity matching failed: {str(e)}"}

        return jsonify({
            "overall": {
                "verdict": overall_verdict,
                "fake_votes": fake_votes,
                "real_votes": real_votes,
                "avg_fake_probability": avg_fake_prob,
                "avg_real_probability": avg_real_prob,
                "total_models": len(results),
            },
            "models": results,
            "errors": errors,
            "identity_match": identity_match,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
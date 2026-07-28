
import os
import json
import sqlite3
import tempfile

import cv2
import numpy as np



_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH     = os.path.join(_BASE_DIR, "users.db")
UPLOADS_DIR = os.path.join(_BASE_DIR, "uploads")

# Must match embedding_worker.py exactly
RECOGNITION_MODEL    = "Facenet512"
DETECTOR_BACKEND     = "opencv"
SIMILARITY_THRESHOLD = 4.0

MAX_PROBE_FRAMES = 15   # how many frames to sample from the video



_deepface_cache = None

def _get_deepface():
    global _deepface_cache
    if _deepface_cache is None:
        try:
            from deepface import DeepFace
            _deepface_cache = DeepFace
            print("[identity_matcher] DeepFace loaded OK")
        except ImportError:
            _deepface_cache = False
            print("[identity_matcher] deepface not installed")
    return _deepface_cache if _deepface_cache is not False else None



def _extract_best_frame(video_path: str) -> str | None:
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[identity_matcher] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 9999  # fallback for streams

    # Sample evenly across the video so we don't just grab the first N frames
    step = max(1, total_frames // MAX_PROBE_FRAMES)

    candidates = []   
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue

        # Use the largest face's crop to score sharpness, but keep the full frame
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop       = gray[y:y+h, x:x+w]
        sharpness  = cv2.Laplacian(crop, cv2.CV_64F).var()
        candidates.append((sharpness, frame.copy()))

        if len(candidates) >= MAX_PROBE_FRAMES:
            break

    cap.release()

    if not candidates:
        print("[identity_matcher] No faces found in video")
        return None

    # Pick the sharpest frame
    best_frame = max(candidates, key=lambda t: t[0])[1]

    # Save as temp JPEG — DeepFace will do its own detection + alignment on this
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    tmp.close()
    print(f"[identity_matcher] Best probe frame saved ({best_frame.shape[1]}x{best_frame.shape[0]})")
    return tmp.name


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_users_with_embeddings():
    """
    Returns (with_embeddings, without_embeddings).
    with_embeddings    — have stored face_embedding in DB (fast dot-product path)
    without_embeddings — no stored embedding (live DeepFace.represent fallback)
    """
    if not os.path.exists(DB_PATH):
        print(f"[identity_matcher] DB not found: {DB_PATH}")
        return [], []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, username, email, face, face_embedding, embedding_status "
            "FROM users WHERE face IS NOT NULL"
        ).fetchall()
        conn.close()

        with_emb, without_emb = [], []
        for r in rows:
            d = dict(r)
            if d.get("face_embedding"):
                try:
                    d["embedding_vector"] = json.loads(d["face_embedding"])
                    with_emb.append(d)
                except Exception:
                    without_emb.append(d)
            else:
                without_emb.append(d)

        print(f"[identity_matcher] {len(with_emb)} with embeddings, "
              f"{len(without_emb)} without")
        return with_emb, without_emb

    except Exception as e:
        print(f"[identity_matcher] DB error: {e}")
        return [], []



def cosine_similarity_pct(a: list, b: list) -> float:
    va = np.array(a, dtype="float32")
    vb = np.array(b, dtype="float32")
    va = va / (np.linalg.norm(va) + 1e-10)
    vb = vb / (np.linalg.norm(vb) + 1e-10)
    return round(float(max(0.0, np.dot(va, vb))) * 100, 1)



def match_identity(video_path: str) -> dict:
    DeepFace = _get_deepface()
    if DeepFace is None:
        return {"error": "deepface not installed — run: pip install deepface tf-keras"}

    with_emb, without_emb = get_users_with_embeddings()
    if not with_emb and not without_emb:
        return {"error": "No registered users in database"}

    # ── Step 1: extract best frame from video ────────────────────────────────
    probe_path = _extract_best_frame(video_path)
    if probe_path is None:
        return {"error": "No face detected in video"}

    best_match = None
    best_sim   = -1.0

    try:
        try:
            probe_result = DeepFace.represent(
                img_path=probe_path,
                model_name=RECOGNITION_MODEL,    # Facenet512
                detector_backend=DETECTOR_BACKEND,  # opencv
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            print(f"[identity_matcher] Probe embedding failed: {e}")
            return {"error": f"Could not generate probe embedding: {e}"}

        if not probe_result:
            return {"error": "No face detected in probe frame"}

        probe_vec = probe_result[0]["embedding"]
        print(f"[identity_matcher] Probe OK — {len(probe_vec)} dims")

        
        for user in with_emb:
            sim = cosine_similarity_pct(probe_vec, user["embedding_vector"])
            print(f"[identity_matcher] vs {user['username']}: {sim}%")
            if sim > best_sim:
                best_sim   = sim
                best_match = _make_match_dict(user, sim, "stored_embedding")

       
        for user in without_emb:
            gallery_path = os.path.join(UPLOADS_DIR, user["face"])
            if not os.path.exists(gallery_path):
                print(f"[identity_matcher] Missing image: {gallery_path}")
                continue
            try:
                g = DeepFace.represent(
                    img_path=gallery_path,
                    model_name=RECOGNITION_MODEL,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )
                if not g:
                    continue
                sim = cosine_similarity_pct(probe_vec, g[0]["embedding"])
                print(f"[identity_matcher] (live) vs {user['username']}: {sim}%")
                if sim > best_sim:
                    best_sim   = sim
                    best_match = _make_match_dict(user, sim, "live_represent")
            except Exception as e:
                print(f"[identity_matcher] Live error for {user['username']}: {e}")

    finally:
        if probe_path and os.path.exists(probe_path):
            try:
                os.unlink(probe_path)
            except Exception:
                pass

    if best_match is None:
        return {"error": "Could not compare with any registered user"}

    best_match["matched"] = best_sim >= SIMILARITY_THRESHOLD

    return {
        "matched":           best_match["matched"],
        "best_match":        best_match if best_match["matched"] else None,
        "closest_candidate": best_match,
        "threshold_used":    SIMILARITY_THRESHOLD,
        "model_used":        RECOGNITION_MODEL,
    }


def _make_match_dict(user: dict, sim: float, method: str) -> dict:
    return {
        "user_id":       user["id"],
        "username":      user["username"],
        "email":         user["email"],
        "face_filename": user["face"],
        "similarity":    sim,
        "matched":       False,
        "method":        method,
    }
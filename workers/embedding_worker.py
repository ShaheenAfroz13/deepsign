"""
embedding_worker.py
--------------------
Called by server.js after a new user registers.
 
Usage:
    python embedding_worker.py <user_id> <image_path>
 
What it does:
    1. Generates a Facenet512 face embedding from the uploaded image
    2. Stores it as a JSON blob in the `face_embedding` column of users.db
 
Install deps (once):
    pip install deepface tf-keras opencv-python-headless
"""
 
import sys
import os
import json
import sqlite3
import numpy as np
 
DB_PATH = "users.db"
RECOGNITION_MODEL = "Facenet512"  # must match the model used in app.py
 
 
def ensure_embedding_column():
    """Add face_embedding column to users table if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN face_embedding TEXT")
        conn.commit()
        print("[DB] Added face_embedding column")
    except sqlite3.OperationalError:
        pass  # Column already exists — fine
    finally:
        conn.close()
 
 
def generate_embedding(image_path: str) -> list | None:
    """Return a flat list of floats (the face embedding), or None on failure."""
    try:
        from deepface import DeepFace
    except ImportError:
        print("[ERROR] deepface not installed. Run: pip install deepface tf-keras")
        return None
 
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=RECOGNITION_MODEL,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )
        # result is a list of dicts; take the first (most prominent) face
        if result and len(result) > 0:
            embedding = result[0]["embedding"]
            return embedding  # already a plain Python list of floats
        else:
            print("[WARN] No face found in image")
            return None
    except Exception as e:
        print(f"[ERROR] DeepFace.represent failed: {e}")
        return None
 
 
def store_embedding(user_id: int, embedding: list):
    """Write the embedding JSON into users.db for the given user_id."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "UPDATE users SET face_embedding = ? WHERE id = ?",
            (json.dumps(embedding), user_id),
        )
        conn.commit()
        print(f"[OK] Embedding stored for user_id={user_id} ({len(embedding)} dims)")
    except Exception as e:
        print(f"[ERROR] DB write failed: {e}")
        sys.exit(1)
    finally:
        conn.close()
 
 
def main():
    if len(sys.argv) != 3:
        print("Usage: python embedding_worker.py <user_id> <image_path>")
        sys.exit(1)
 
    user_id = int(sys.argv[1])
    image_path = sys.argv[2]
 
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
 
    ensure_embedding_column()
 
    print(f"[INFO] Generating embedding for user_id={user_id}, image={image_path}")
    embedding = generate_embedding(image_path)
 
    if embedding is None:
        print("[WARN] Could not generate embedding — user registered without embedding")
        sys.exit(0)  # Exit 0 so server.js doesn't treat this as a hard failure
 
    store_embedding(user_id, embedding)
 
 
if __name__ == "__main__":
    main()
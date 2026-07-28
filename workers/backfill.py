import sqlite3
import subprocess
import os
import sys
#for filling embeddings for user faces
DB_PATH = "users.db"
UPLOADS_DIR = "uploads"

conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT id, face FROM users WHERE face IS NOT NULL"
).fetchall()
conn.close()

if not rows:
    print("No users found — nothing to backfill")
else:
    for user_id, face in rows:
        image_path = os.path.join(UPLOADS_DIR, face)
        if not os.path.exists(image_path):
            print(f"[SKIP] Image not found: {image_path}")
            continue

        print(f"[PROCESSING] user_id={user_id}, image={image_path}")
        result = subprocess.run([
            sys.executable,
            "embedding_worker.py",
            str(user_id),
            image_path
        ])

        # Update embedding_status based on whether the worker succeeded
        status = "done" if result.returncode == 0 else "failed"
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE users SET embedding_status = ? WHERE id = ?",
            (status, user_id)
        )
        conn.commit()
        conn.close()
        print(f"[STATUS] user_id={user_id} → {status}")

print("\nDone — verify with: sqlite3 users.db \"SELECT id, username, embedding_status FROM users;\"")

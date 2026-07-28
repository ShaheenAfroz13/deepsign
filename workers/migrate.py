import sqlite3

conn = sqlite3.connect("users.db")

try:
    conn.execute("ALTER TABLE users ADD COLUMN face_embedding TEXT")
    print("[OK] Added face_embedding column")
except sqlite3.OperationalError:
    print("[SKIP] face_embedding column already exists")

try:
    conn.execute("ALTER TABLE users ADD COLUMN embedding_status TEXT DEFAULT 'pending'")
    print("[OK] Added embedding_status column")
except sqlite3.OperationalError:
    print("[SKIP] embedding_status column already exists")

conn.execute("UPDATE users SET embedding_status = 'pending' WHERE embedding_status IS NULL")
conn.commit()
conn.close()

print("Migration complete — now run: python backfill.py")
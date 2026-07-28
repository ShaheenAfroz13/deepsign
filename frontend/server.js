console.log("SERVER STARTING")
const express  = require("express")
const sqlite3  = require("sqlite3").verbose()
const multer   = require("multer")
const bcrypt   = require("bcryptjs")
const fs       = require("fs")
const path     = require("path")
const { spawn } = require("child_process")   
const app = express()

app.use(express.json())
app.use(express.static("public"))
app.use("/uploads", express.static("uploads"))

if (!fs.existsSync("uploads")) fs.mkdirSync("uploads")

const db = new sqlite3.Database("users.db")

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      username       TEXT,
      email          TEXT UNIQUE,
      password       TEXT,
      face           TEXT,
      face_embedding TEXT,           -- JSON array of floats, written by Python
      embedding_status TEXT DEFAULT 'pending'  -- pending | done | failed
    )
  `)
})

// ── Multer ──────────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename:    (req, file, cb) => cb(null, Date.now() + "-" + file.originalname)
})
const upload = multer({ storage })


function spawnEmbeddingWorker(userId, imagePath) {
  const workerPath = path.resolve(__dirname, "embedding_worker.py")
  const absImagePath = path.resolve(__dirname, imagePath)

 
  const python = process.env.VIRTUAL_ENV
  ? require("path").join(process.env.VIRTUAL_ENV, "Scripts", "python.exe")
  : "python"

  const worker = spawn(python, [workerPath, String(userId), absImagePath], {
    detached: true,   
    stdio:    "pipe"  
  })

  worker.stdout.on("data", d => console.log(`[embedding_worker] ${d.toString().trim()}`))
  worker.stderr.on("data", d => console.error(`[embedding_worker ERROR] ${d.toString().trim()}`))

  worker.on("close", code => {
    const status = code === 0 ? "done" : "failed"
    console.log(`[embedding_worker] Exited ${code} for user_id=${userId} → status=${status}`)

    
    db.run(
      "UPDATE users SET embedding_status = ? WHERE id = ?",
      [status, userId],
      err => { if (err) console.error("[DB] Status update failed:", err.message) }
    )
  })

  worker.unref()  
}



app.post("/register", upload.single("face"), async (req, res) => {
  const { username, email, password } = req.body
  if (!username || !email || !password)
    return res.json({ error: "All fields required" })
  if (!req.file)
    return res.json({ error: "Face image required" })

  const hashed = await bcrypt.hash(password, 10)
  const imagePath = req.file.path  

  db.run(
    `INSERT INTO users (username, email, password, face, embedding_status)
     VALUES (?, ?, ?, ?, 'pending')`,
    [username, email, hashed, req.file.filename],
    function (err) {
      if (err) {
        if (err.message.includes("UNIQUE"))
          return res.json({ error: "Email already registered" })
        return res.json({ error: "Database error" })
      }

      const newUserId = this.lastID

      
      res.json({ success: true, id: newUserId, username })

      spawnEmbeddingWorker(newUserId, imagePath)
    }
  )
})



app.post("/login", (req, res) => {
  const { email, password } = req.body
  if (!email || !password)
    return res.json({ error: "Email and password required" })

  db.get("SELECT * FROM users WHERE email = ?", [email], async (err, user) => {
    if (err)   return res.json({ error: "Database error" })
    if (!user) return res.json({ error: "Invalid credentials" })

    const match = await bcrypt.compare(password, user.password)
    if (!match) return res.json({ error: "Invalid credentials" })

    res.json({ success: true, id: user.id, username: user.username })
  })
})

app.get("/users", (req, res) => {
  db.all(
    "SELECT id, username, email, face, embedding_status FROM users",
    [],
    (err, rows) => {
      if (err) return res.json({ error: "Database error" })
      res.json(rows)
    }
  )
})


app.get("/users/:id/embedding-status", (req, res) => {
  db.get(
    "SELECT embedding_status FROM users WHERE id = ?",
    [req.params.id],
    (err, row) => {
      if (err || !row) return res.json({ error: "User not found" })
      res.json({ status: row.embedding_status })
    }
  )
})


app.listen(8000, () => console.log("Server running on port 8000"))
#localhost-8000 
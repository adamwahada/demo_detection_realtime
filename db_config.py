"""
Database & storage configuration.
All connection settings live here — never hardcoded elsewhere.
"""

# ==========================
# PostgreSQL (Docker container)
# ==========================
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "tracking"
DB_USER     = "tracking_user"
DB_PASSWORD = "changeme"

# Connection string (for tools like DBeaver or psql cli)
# psql -U tracking_user -d tracking -h localhost

# ==========================
# MinIO (local S3 — Docker container)
# ==========================
MINIO_ENDPOINT  = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
MINIO_BUCKET    = "tracking-images"
MINIO_SECURE    = False   # True if using HTTPS

# ==========================
# Stats behaviour
# ==========================
# Whether stats recording is ON by default when the server starts
STATS_ENABLED_DEFAULT = True

# Save a KPI snapshot to stats_snapshots every N packets
# (used for trend charts on the dashboard)
SNAPSHOT_EVERY_N_PACKETS = 50

# Write queue max size — if DB writer falls behind, older events are dropped
# to never block the detection thread. 10k events ≈ ~5MB RAM.
WRITE_QUEUE_MAXSIZE = 10_000

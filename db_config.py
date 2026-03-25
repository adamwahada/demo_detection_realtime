"""Database and async stats recording configuration for demo_detection_realtime."""

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "tracking"
DB_USER = "tracking_user"
DB_PASSWORD = "changeme"

STATS_ENABLED_DEFAULT = False
SNAPSHOT_EVERY_N_PACKETS = 25
WRITE_QUEUE_MAXSIZE = 10000

"""Database and async stats recording configuration for demo_detection_realtime."""

import os

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5434"))
DB_NAME = os.environ.get("DB_NAME", "farine_detection")
DB_USER = os.environ.get("DB_USER", "farine_khomsa")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "aa")

STATS_ENABLED_DEFAULT = False
SNAPSHOT_EVERY_N_PACKETS = 25
WRITE_QUEUE_MAXSIZE = 10000

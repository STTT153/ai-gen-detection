# Gunicorn configuration for production deployment on Apple Silicon (M-series).
#
# Usage (from repo root):
#   source web/venv/bin/activate
#   gunicorn --chdir web --config web/gunicorn.conf.py app:app

import multiprocessing

# Single worker to load the model once and avoid duplicating GPU/MPS memory.
workers = 1
worker_class = "gthread"
threads = 4

bind = "0.0.0.0:8000"

# Graceful restart window — gives in-flight inference requests time to finish.
graceful_timeout = 120
timeout = 120

# Logging
accesslog = "server.log"
errorlog = "-"  # stderr
loglevel = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# Restart workers after this many requests to reclaim any memory drift.
max_requests = 500
max_requests_jitter = 50

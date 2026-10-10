import os

# Gunicorn configuration for Brain Tumor AI
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"
workers = 1
threads = 2
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"

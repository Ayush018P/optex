"""Root entrypoint for Vercel deployment.

Vercel's static scanner requires an explicit `from flask import Flask`
in the file to recognise it as a Flask application.
Dash builds on Flask, so we re-export the underlying Flask server here.
"""
from __future__ import annotations

import sys
import os

# Flask import is required for Vercel's framework detector (static scan)
from flask import Flask  # noqa: F401

# Make sure the repo root is on the path so dashboard.* imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import server

# 'app' must be the Flask WSGI callable — Vercel, Gunicorn, and uWSGI all
# look for a module-level variable named `app`.
app: Flask = server


"""Root entrypoint for Vercel deployment.

Vercel looks for a Flask `app` (or `server`) object in root-level files.
We re-export both names from the actual Dash app so Vercel can find it.
"""
from __future__ import annotations

import sys
import os

# Make sure the repo root is on the path so dashboard.* imports work
sys.path.insert(0, os.path.dirname(__file__))

from dashboard.app import app, server  # noqa: F401 – re-exported for Vercel

# Vercel / WSGI servers look for `app` as the callable
app = server

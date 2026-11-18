"""Qt interface for the face recognition app.

Layering: `app` builds the window, `context` holds the shared state, each page owns
one task, `widgets` holds the reusable pieces, and `theme` holds every colour.
Pages never import each other.
"""

from __future__ import annotations

from .app import MainWindow, run_app
from .context import AppContext

__all__ = ["AppContext", "MainWindow", "run_app"]

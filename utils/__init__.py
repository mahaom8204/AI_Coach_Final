# utils/__init__.py
"""
Utility package for Adaptive English Learning Coach.
Includes:
- roadmap_loader: loads and flattens English_Roadmap.json
- session_state: manages Streamlit session data
"""

from . import roadmap_loader
from . import session_state

__all__ = ["roadmap_loader", "session_state"]

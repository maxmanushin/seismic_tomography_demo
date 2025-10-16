"""
Top-level package for the educational seismotomography project.

The code is intentionally lightweight: it keeps to straight-ray tomography for
the inversion side, analytic head-wave calculations for talking through the
hodograph, and exposes utility functions that the Streamlit UI can reuse.
"""

from .tomography import *  # noqa: F401,F403

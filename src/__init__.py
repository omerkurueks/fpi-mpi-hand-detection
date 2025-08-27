"""
Hand-held Object Inspection Detection System

A modular system for detecting and tracking objects held in hands,
analyzing inspection behavior through motion analysis and finite state machines.
"""

__version__ = "1.0.0"
__author__ = "FPI-MPI Hand Detection Team"

from .pipeline import Pipeline
from .config import Config

__all__ = ["Pipeline", "Config"]

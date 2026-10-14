#!/usr/bin/env python3
"""
FreightTiger Smart Shipping Cost Intelligence Assistant
Main Entry Point
"""

import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    main()

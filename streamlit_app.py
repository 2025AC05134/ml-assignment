"""Streamlit Cloud entry point.

Streamlit Community Cloud looks for `streamlit_app.py` by default. This
module simply re-exports `main` from `app.py` so both entry points work.
"""
from app import main

if __name__ == "__main__":
    main()

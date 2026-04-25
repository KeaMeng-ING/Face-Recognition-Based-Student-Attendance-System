"""
Project entrypoint.

Runs the attendance application pipeline (face recognition + anti-spoofing).
"""

from app import main as run_attendance_app


if __name__ == "__main__":
    run_attendance_app()

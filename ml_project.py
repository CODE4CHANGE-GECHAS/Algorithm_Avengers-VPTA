import sys

exercise = sys.argv[1] if len(sys.argv) > 1 else None
if exercise not in exercise_options:
    print("❌ Invalid or no exercise selected. Exiting.")
    exit()

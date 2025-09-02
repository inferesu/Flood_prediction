import torch
import os
from xanfis import GdAnfisRegressor

MODEL_FILE = 'bug_report_model.json'

def attempt_to_load_model():
    """Attempts to load the saved model, demonstrating the bug."""
    print(f"--- Attempting to load model from '{MODEL_FILE}' ---")

    if not os.path.exists(MODEL_FILE):
        print("Model file not found. Please run 'minimal_training_script.py' first.")
        return

    try:
        # 1. Create a new, empty model object.
        model = GdAnfisRegressor()
        # 2. Use the library's dedicated function to load the saved model.
        # THIS IS THE LINE THAT WILL FAIL.
        model.load_model(MODEL_FILE)

        print("Model loaded without error.") # This line will likely not be reached.

    except Exception as e:
        print(f"\nFAILED TO LOAD MODEL. This demonstrates the bug.")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    attempt_to_load_model()

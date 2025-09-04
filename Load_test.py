# final_minimal_loading_script.py
# This script uses the developer's official method to load the saved model.

import os
from xanfis import GdAnfisRegressor

# --- Configuration ---
# This filename now matches the one in the saving script.
# MODEL_FILENAME = 'bug_report_model.pkl'

def attempt_to_load_model():
    """Loads the saved model using the developer's official method."""

    # if not os.path.exists(MODEL_FILENAME):
    #     print(f"Model file not found.")
    #     return

    try:
        cls = GdAnfisRegressor
        model = cls.load_model(load_path=".", filename="bug_report_model.pkl")

        print("\n✅ SUCCESS: Model loaded without error.")
        print(f"   Model Type: {type(model)}")

    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL.")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    attempt_to_load_model()

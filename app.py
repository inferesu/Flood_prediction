import logging
import time
import threading
from dotenv import load_dotenv
from flask import Flask

from routes.api import api_bp
from scheduler import start_scheduler
from config.rivers import RIVER_CONFIGS
from services.predictor import run_prediction_job

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flood_app")


def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app


def run_startup_jobs():
    """Run all prediction jobs in parallel on startup."""

    def _run_one(cfg):
        try:
            run_prediction_job(cfg)
        except Exception as e:
            logger.error(f"Startup job failed for {cfg['display_name']}: {e}")

    def _startup():
        logger.info("=== Running startup prediction jobs ===")
        time.sleep(2)

        threads = [
            threading.Thread(target=_run_one, args=(cfg,), daemon=True)
            for cfg in RIVER_CONFIGS.values()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.info("=== Startup prediction jobs complete ===")

    threading.Thread(target=_startup, daemon=True).start()


app = create_app()
start_scheduler()
run_startup_jobs()

if __name__ == "__main__":
    app.run(debug=True, port=5002, use_reloader=False)

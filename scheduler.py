"""Background scheduler for periodic prediction jobs."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from config.rivers import RIVER_CONFIGS
from services.predictor import run_prediction_job

logger = logging.getLogger("flood_app")

_scheduler = BackgroundScheduler()


def start_scheduler():
    """Register and start all prediction cron jobs."""
    for key, cfg in RIVER_CONFIGS.items():
        _scheduler.add_job(
            func=run_prediction_job,
            args=(cfg,),
            trigger="cron",
            minute="05",
            id=f"predict_{key}",
            replace_existing=True,
        )
        logger.info(f"Scheduled prediction job for: {cfg['display_name']}")

    _scheduler.start()
    logger.info(f"Scheduler started with {len(RIVER_CONFIGS)} jobs.")

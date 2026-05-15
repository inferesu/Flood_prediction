"""Gemini-powered AI summary of current flood conditions."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from config.rivers import RIVER_CONFIGS

logger = logging.getLogger("flood_app")


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        _client = genai.Client(api_key=api_key)
    return _client


def _determine_risk(level_val, risk_levels: list) -> str:
    if not isinstance(level_val, (int, float)):
        return "UNKNOWN"
    r1, r2, r3 = risk_levels
    if level_val > r3:
        return "SEVERE"
    elif level_val > r2:
        return "HIGH"
    elif level_val > r1:
        return "MODERATE"
    return "LOW"


def generate_summary() -> dict:
    river_lines = []

    for key, cfg in RIVER_CONFIGS.items():
        path = cfg["predictions_log_path"]
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                log = json.load(f)
            last_entry = log[sorted(log.keys())[-1]]
            actual = last_entry.get("actual", "—")
            h = last_entry.get("horizons", {})
            h1d = h.get("1d", {}).get("level", "—")
            h3d = h.get("3d", {}).get("level", "—")
            h5d = h.get("5d", {}).get("level", "—")

            level_val = h1d if isinstance(h1d, (int, float)) else actual
            risk = _determine_risk(level_val, cfg["risk_levels"])

            river_lines.append(
                f"- {cfg['display_name']}: current={actual} cm, "
                f"1d={h1d} cm, 3d={h3d} cm, 5d={h5d} cm, risk={risk}"
            )
        except Exception:
            continue

    if not river_lines:
        return {"summary": "No river data available yet.", "river_count": 0}

    prompt = (
        "You are a flood risk analyst. Based on the following real-time water level "
        "readings and forecasts for Lithuanian rivers, write a concise 3-4 sentence "
        "summary for emergency management personnel. Highlight which rivers pose the "
        "highest immediate risk, any concerning trends, and which are currently safe. "
        "Use plain language, no bullet points, no markdown.\n\n"
        "River data:\n" + "\n".join(river_lines)
    )

    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return {
        "summary": response.text.strip(),
        "river_count": len(river_lines),
        "generated_at": datetime.now().isoformat(),
    }

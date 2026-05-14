import json
import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai as genai


import joblib
import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

from anfis.anfis import AnfisNet
from anfis.membership import BellMembFunc



load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
warnings.filterwarnings("ignore")
logger = logging.getLogger('flood_app')
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

K_DECAY = 0.85

# ---------------------------------------------------------------------------
# INFRASTRUCTURE DATA
# ---------------------------------------------------------------------------
INFRASTRUCTURE_DATA = {
    "minija": [
        {"name": "Klaipėda University Hospital", "type": "hospital",  "lat": 55.7067, "lon": 21.1443},
        {"name": "Priekulė Emergency Shelter",   "type": "shelter",   "lat": 55.5446, "lon": 21.3295},
        {"name": "Kintai Bridge",                "type": "bridge",    "lat": 55.3500, "lon": 21.2500},
        {"name": "Minija Dam",                   "type": "dam",       "lat": 55.6200, "lon": 21.2800},
        {"name": "Priekulė Primary School",      "type": "school",    "lat": 55.5566, "lon": 21.3115},
        {"name": "Klaipėda Power Station",       "type": "power",     "lat": 55.7100, "lon": 21.1300},
        {"name": "Gargždai Hospital",            "type": "hospital",  "lat": 55.7222, "lon": 21.3889},
        {"name": "Minija Fire Station",          "type": "shelter",   "lat": 55.3531, "lon": 21.2531},
    ],
    "dane": [
        {"name": "Klaipėda University Hospital", "type": "hospital",  "lat": 55.7067, "lon": 21.1443},
        {"name": "Klaipėda City Shelter",        "type": "shelter",   "lat": 55.7200, "lon": 21.1500},
        {"name": "Danė Railway Bridge",          "type": "bridge",    "lat": 55.7050, "lon": 21.1350},
        {"name": "Klaipėda Power Plant",         "type": "power",     "lat": 55.7150, "lon": 21.1200},
        {"name": "Vitės School",                 "type": "school",    "lat": 55.7300, "lon": 21.1600},
        {"name": "Danė Fire Station",            "type": "shelter",   "lat": 55.7080, "lon": 21.1420},
    ],
    "kartena": [
        {"name": "Kartena Primary School",       "type": "school",    "lat": 55.9200, "lon": 21.4800},
        {"name": "Kartena Community Shelter",    "type": "shelter",   "lat": 55.9150, "lon": 21.4750},
        {"name": "Kartena Bridge",               "type": "bridge",    "lat": 55.9096, "lon": 21.4679},
        {"name": "Kretinga Hospital",            "type": "hospital",  "lat": 55.8833, "lon": 21.2333},
        {"name": "Kartena Fire Station",         "type": "shelter",   "lat": 55.9100, "lon": 21.4700},
        {"name": "Minija–Kartena Dam",           "type": "dam",       "lat": 55.9050, "lon": 21.4600},
    ],
    "dane_kretinga": [
        {"name": "Kretinga Hospital",            "type": "hospital",  "lat": 55.8833, "lon": 21.2333},
        {"name": "Kretinga Community Shelter",   "type": "shelter",   "lat": 55.8850, "lon": 21.2400},
        {"name": "Danė–Kretinga Bridge",         "type": "bridge",    "lat": 55.8606, "lon": 21.2206},
        {"name": "Kretinga Power Station",       "type": "power",     "lat": 55.8750, "lon": 21.2100},
        {"name": "Kretinga Primary School",      "type": "school",    "lat": 55.8900, "lon": 21.2350},
        {"name": "Kretinga Fire Station",        "type": "shelter",   "lat": 55.8620, "lon": 21.2250},
    ],
    # ── NEW ──────────────────────────────────────────────────────────────
    "kanalas_lankupiu": [
        {"name": "Lankupiai Community Shelter",  "type": "shelter",   "lat": 55.6550, "lon": 21.3600},
        {"name": "Lankupiai Bridge (Klaipėda canal)", "type": "bridge","lat": 55.6500, "lon": 21.3550},
        {"name": "Klaipėda University Hospital", "type": "hospital",  "lat": 55.7067, "lon": 21.1443},
        {"name": "Lankupiai Fire Station",       "type": "shelter",   "lat": 55.6520, "lon": 21.3580},
        {"name": "Lankupiai School",             "type": "school",    "lat": 55.6570, "lon": 21.3620},
        {"name": "Klaipėda Power Plant",         "type": "power",     "lat": 55.7150, "lon": 21.1200},
    ],
    "lankupiu_minija": [
        {"name": "Lankupiai Community Shelter",  "type": "shelter",   "lat": 55.6550, "lon": 21.3900},
        {"name": "Lankupiai Bridge (Minija)",    "type": "bridge",    "lat": 55.6500, "lon": 21.3850},
        {"name": "Gargždai Hospital",            "type": "hospital",  "lat": 55.7222, "lon": 21.3889},
        {"name": "Lankupiai Fire Station",       "type": "shelter",   "lat": 55.6520, "lon": 21.3870},
        {"name": "Lankupiai Primary School",     "type": "school",    "lat": 55.6570, "lon": 21.3920},
        {"name": "Minija Dam",                   "type": "dam",       "lat": 55.6200, "lon": 21.2800},
    ],
    "marios_birstonas": [
        {"name": "Birštonas Hospital",           "type": "hospital",  "lat": 54.6135, "lon": 24.0336},
        {"name": "Birštonas Community Shelter",  "type": "shelter",   "lat": 54.6150, "lon": 24.0350},
        {"name": "Birštonas Bridge",             "type": "bridge",    "lat": 54.6120, "lon": 24.0320},
        {"name": "Birštonas Primary School",     "type": "school",    "lat": 54.6160, "lon": 24.0360},
        {"name": "Birštonas Fire Station",       "type": "shelter",   "lat": 54.6130, "lon": 24.0310},
        {"name": "Marios Power Station",         "type": "power",     "lat": 54.6100, "lon": 24.0280},
    ],
    "nemunas_nemajunai": [
        {"name": "Nemajūnai Community Shelter",  "type": "shelter",   "lat": 54.5542, "lon": 24.0720},
        {"name": "Nemajūnai Bridge",             "type": "bridge",    "lat": 54.5530, "lon": 24.0710},
        {"name": "Alytus Hospital",              "type": "hospital",  "lat": 54.4000, "lon": 24.0460},
        {"name": "Nemajūnai Primary School",     "type": "school",    "lat": 54.5560, "lon": 24.0730},
        {"name": "Nemajūnai Fire Station",       "type": "shelter",   "lat": 54.5520, "lon": 24.0700},
        {"name": "Nemunas Power Station",        "type": "power",     "lat": 54.5500, "lon": 24.0680},
    ],
    "nemunas_darsuniskis": [
        {"name": "Darsūniškis Community Shelter", "type": "shelter",  "lat": 54.3700, "lon": 24.2100},
        {"name": "Darsūniškis Bridge",            "type": "bridge",   "lat": 54.3690, "lon": 24.2090},
        {"name": "Kaišiadorys Hospital",          "type": "hospital", "lat": 54.8650, "lon": 24.4500},
        {"name": "Darsūniškis Primary School",    "type": "school",   "lat": 54.3710, "lon": 24.2110},
        {"name": "Darsūniškis Fire Station",      "type": "shelter",  "lat": 54.3680, "lon": 24.2080},
        {"name": "Nemunas Power Station",         "type": "power",    "lat": 54.3660, "lon": 24.2060},
    ],
    "nemunas_lampedziai": [
        {"name": "Lampėdžiai Community Shelter", "type": "shelter", "lat": 54.9064, "lon": 23.8176},
        {"name": "Lampėdžiai Bridge", "type": "bridge", "lat": 54.9050, "lon": 23.8160},
        {"name": "Kaunas University Hospital", "type": "hospital", "lat": 54.9027, "lon": 23.9096},
        {"name": "Lampėdžiai Primary School", "type": "school", "lat": 54.9075, "lon": 23.8190},
        {"name": "Lampėdžiai Fire Station", "type": "shelter", "lat": 54.9040, "lon": 23.8150},
        {"name": "Nemunas Power Station", "type": "power", "lat": 54.9020, "lon": 23.8130},
    ],
    "vilnia_vilnius": [
        {"name": "Vilnius University Hospital", "type": "hospital", "lat": 54.6872, "lon": 25.2797},
        {"name": "Vilnius City Shelter", "type": "shelter", "lat": 54.6890, "lon": 25.2750},
        {"name": "Vilnia Bridge", "type": "bridge", "lat": 54.6860, "lon": 25.2810},
        {"name": "Vilnius Primary School", "type": "school", "lat": 54.6900, "lon": 25.2820},
        {"name": "Vilnius Fire Station", "type": "shelter", "lat": 54.6850, "lon": 25.2780},
        {"name": "Vilnius Power Station", "type": "power", "lat": 54.6840, "lon": 25.2760},
    ],
    "neris_vilnius": [
        {"name": "Vilnius University Hospital", "type": "hospital", "lat": 54.6872, "lon": 25.2797},
        {"name": "Vilnius City Shelter", "type": "shelter", "lat": 54.6890, "lon": 25.2750},
        {"name": "Neris Bridge (Vilnius)", "type": "bridge", "lat": 54.6880, "lon": 25.2780},
        {"name": "Vilnius Primary School", "type": "school", "lat": 54.6900, "lon": 25.2820},
        {"name": "Vilnius Fire Station", "type": "shelter", "lat": 54.6850, "lon": 25.2760},
        {"name": "Vilnius Power Station", "type": "power", "lat": 54.6840, "lon": 25.2740},
    ],

}

# ---------------------------------------------------------------------------
# MONITORING STATIONS
# ---------------------------------------------------------------------------
MONITORING_STATIONS = {
    "minija": [
        {"code": "priekules-vms",            "name": "Priekulė",              "lat": 55.549907, "lon": 21.330332, "is_main": True},
    ],
    "dane": [
        {"code": "klaipedos-vms",            "name": "Klaipėda",              "lat": 55.755884, "lon": 21.135223, "is_main": True},
    ],
    "kartena": [
        {"code": "kartenos-vms",             "name": "Kartena",               "lat": 55.909629, "lon": 21.467874, "is_main": True},
    ],
    "dane_kretinga": [
        {"code": "kretingos-vms",            "name": "Kretinga",              "lat": 55.860562, "lon": 21.220636, "is_main": True},
    ],
    # ── NEW ──────────────────────────────────────────────────────────────
    "kanalas_lankupiu": [
        {"code": "lankupiu-klaipedos-vms",   "name": "Lankupiai (Klaipėda canal)", "lat": 55.6500, "lon": 21.3550, "is_main": True},
    ],
    "lankupiu_minija": [
        {"code": "lankupiu-minija-vms",      "name": "Lankupiai (Minija)",    "lat": 55.6500, "lon": 21.3850, "is_main": True},
    ],
    "marios_birstonas": [
        {"code": "birstono-vms", "name": "Birštonas", "lat": 54.613520, "lon": 24.033594, "is_main": True},
    ],
    "nemunas_nemajunai": [
        {"code": "nemajunu-vms", "name": "Nemajūnai", "lat": 54.554227, "lon": 24.072006, "is_main": True},
    ],
    "nemunas_darsuniskis": [
        {"code": "darsuniskio-vms", "name": "Darsūniškis", "lat": 54.3700, "lon": 24.2100, "is_main": True},
    ],
    "nemunas_lampedziai": [
        {"code": "lampedziu-vms", "name": "Lampėdžiai", "lat": 54.906414, "lon": 23.817574, "is_main": True},
    ],
    "vilnia_vilnius": [
        {"code": "vilniaus-vilnia-vms", "name": "Vilnius (Vilnia)", "lat": 54.679215, "lon": 25.294857, "is_main": True},
    ],
    "neris_vilnius": [
        {"code": "vilniaus-neris-vms", "name": "Vilnius (Neris)", "lat": 54.6872, "lon": 25.2797, "is_main": True},
    ],

}

# ---------------------------------------------------------------------------
# RIVER CONFIGS
# ---------------------------------------------------------------------------
MINIJA_CONFIG = {
    "name": "minija",
    "display_name": "Minija (Priekulė)",
    "data_file": "live_data_complex.csv",
    "predictions_log_path": "predictions_log.json",
    "hydro_station": "priekules-vms",
    "lat": 55.5546,
    "lon": 21.3195,
    "meteo_stations": ["klaipedos-ams", "vezaiciu-ams"],
    "risk_levels": [250, 400, 550],
    "horizons": {
        "1d": {
            "model":      "anfis_model.pth",
            "scaler_x":   "scaler_X.pkl",
            "scaler_y":   "scaler_Y.pkl",
            "config":     "training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "anfis_model_3d.pth",
            "scaler_x":   "scaler_X_3d.pkl",
            "scaler_y":   "scaler_y_3d.pkl",
            "config":     "training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "anfis_model_5d.pth",
            "scaler_x":   "scaler_X_5d.pkl",
            "scaler_y":   "scaler_y_5d.pkl",
            "config":     "training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}

DANE_CONFIG = {
    "name": "dane",
    "display_name": "Danė (Klaipėda)",
    "data_file": "live_data_dane.csv",
    "predictions_log_path": "predictions_log_dane.json",
    "hydro_station": "klaipedos-vms",
    "lat": 55.755884,
    "lon": 21.135223,
    "meteo_stations": ["klaipedos-ams"],
    "risk_levels": [150, 280, 400],
    "horizons": {
        "1d": {
            "model":      "dane_anfis_model.pth",
            "scaler_x":   "dane_scaler_X.pkl",
            "scaler_y":   "dane_scaler_y.pkl",
            "config":     "dane_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "dane_anfis_model_3d.pth",
            "scaler_x":   "dane_scaler_X_3d.pkl",
            "scaler_y":   "dane_scaler_y_3d.pkl",
            "config":     "dane_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "dane_anfis_model_5d.pth",
            "scaler_x":   "dane_scaler_X_5d.pkl",
            "scaler_y":   "dane_scaler_y_5d.pkl",
            "config":     "dane_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}

KARTENA_CONFIG = {
    "name": "kartena",
    "display_name": "Minija (Kartena)",
    "data_file": "live_data_kartena.csv",
    "predictions_log_path": "predictions_log_kartena.json",
    "hydro_station": "kartenos-vms",
    "lat": 55.909629,
    "lon": 21.467874,
    "meteo_stations": ["kretingos-ams", "vezaiciu-ams"],
    "risk_levels": [200, 350, 500],
    "horizons": {
        "1d": {
            "model":      "kartena_anfis_model.pth",
            "scaler_x":   "kartena_scaler_X.pkl",
            "scaler_y":   "kartena_scaler_y.pkl",
            "config":     "kartena_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {                                               # ← NEW
            "model":      "kartena_anfis_model_3d.pth",
            "scaler_x":   "kartena_scaler_X_3d.pkl",
            "scaler_y":   "kartena_scaler_y_3d.pkl",
            "config":     "kartena_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {                                               # ← NEW
            "model":      "kartena_anfis_model_5d.pth",
            "scaler_x":   "kartena_scaler_X_5d.pkl",
            "scaler_y":   "kartena_scaler_y_5d.pkl",
            "config":     "kartena_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}

DANE_KRETINGA_CONFIG = {
    "name": "dane_kretinga",
    "display_name": "Danė (Kretinga)",
    "data_file": "live_data_dane_kretinga.csv",
    "predictions_log_path": "predictions_log_dane_kretinga.json",
    "hydro_station": "kretingos-vms",
    "lat": 55.860562,
    "lon": 21.220636,
    "meteo_stations": ["kretingos-ams"],
    "risk_levels": [150, 280, 400],
    "horizons": {
        "1d": {
            "model":      "dane_kretinga_anfis_model.pth",
            "scaler_x":   "dane_kretinga_scaler_X.pkl",
            "scaler_y":   "dane_kretinga_scaler_y.pkl",
            "config":     "dane_kretinga_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "dane_kretinga_anfis_model_3d.pth",
            "scaler_x":   "dane_kretinga_scaler_X_3d.pkl",
            "scaler_y":   "dane_kretinga_scaler_y_3d.pkl",
            "config":     "dane_kretinga_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "dane_kretinga_anfis_model_5d.pth",
            "scaler_x":   "dane_kretinga_scaler_X_5d.pkl",
            "scaler_y":   "dane_kretinga_scaler_y_5d.pkl",
            "config":     "dane_kretinga_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}

# ── NEW ──────────────────────────────────────────────────────────────────
KANALAS_LANKUPIU_CONFIG = {
    "name": "kanalas_lankupiu",
    "display_name": "Kanalas (Lankupiai)",
    "data_file": "live_data_kanalas_lankupiu.csv",
    "predictions_log_path": "predictions_log_kanalas_lankupiu.json",
    "hydro_station": "lankupiu-klaipedos-vms",
    "lat": 55.6500,
    "lon": 21.3550,
    "meteo_stations": ["ventes-ams", "klaipedos-ams"],
    "risk_levels": [150, 280, 400],
    "horizons": {
        "1d": {
            "model":      "kanalas_lankupiu_anfis_model.pth",
            "scaler_x":   "kanalas_lankupiu_scaler_X.pkl",
            "scaler_y":   "kanalas_lankupiu_scaler_y.pkl",
            "config":     "kanalas_lankupiu_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {                                                    # ← NEW
            "model":      "kanalas_lankupiu_anfis_model_3d.pth",
            "scaler_x":   "kanalas_lankupiu_scaler_X_3d.pkl",
            "scaler_y":   "kanalas_lankupiu_scaler_y_3d.pkl",
            "config":     "kanalas_lankupiu_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {                                                    # ← NEW
            "model":      "kanalas_lankupiu_anfis_model_5d.pth",
            "scaler_x":   "kanalas_lankupiu_scaler_X_5d.pkl",
            "scaler_y":   "kanalas_lankupiu_scaler_y_5d.pkl",
            "config":     "kanalas_lankupiu_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
LANKUPIU_MINIJA_CONFIG = {
    "name": "lankupiu_minija",
    "display_name": "Minija (Lankupiai)",
    "data_file": "live_data_lankupiu_minija.csv",
    "predictions_log_path": "predictions_log_lankupiu_minija.json",
    "hydro_station": "lankupiu-minija-vms",
    "lat": 55.6500,
    "lon": 21.3850,
    "meteo_stations": ["ventes-ams", "klaipedos-ams"],
    "risk_levels": [200, 350, 500],
    "horizons": {
        "1d": {
            "model":      "lankupiu_minija_anfis_model.pth",
            "scaler_x":   "lankupiu_minija_scaler_X.pkl",
            "scaler_y":   "lankupiu_minija_scaler_y.pkl",
            "config":     "lankupiu_minija_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {                                                    # ← NEW
            "model":      "lankupiu_minija_anfis_model_3d.pth",
            "scaler_x":   "lankupiu_minija_scaler_X_3d.pkl",
            "scaler_y":   "lankupiu_minija_scaler_y_3d.pkl",
            "config":     "lankupiu_minija_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {                                                    # ← NEW
            "model":      "lankupiu_minija_anfis_model_5d.pth",
            "scaler_x":   "lankupiu_minija_scaler_X_5d.pkl",
            "scaler_y":   "lankupiu_minija_scaler_y_5d.pkl",
            "config":     "lankupiu_minija_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },

    },

}
# ---------------------------------------------------------------------------
# NEW CONFIG
# ---------------------------------------------------------------------------
MARIOS_BIRSTONAS_CONFIG = {
    "name": "marios_birstonas",
    "display_name": "Marios (Birštonas)",
    "data_file": "live_data_marios_birstonas.csv",
    "predictions_log_path": "predictions_log_marios_birstonas.json",
    "hydro_station": "birstono-vms",
    "lat": 54.613520,
    "lon": 24.033594,
    "meteo_stations": ["birstono-ams"],
    "risk_levels": [200, 350, 500],
    "horizons": {
        "1d": {
            "model":      "marios_birstonas_anfis_model.pth",
            "scaler_x":   "marios_birstonas_scaler_X.pkl",
            "scaler_y":   "marios_birstonas_scaler_y.pkl",
            "config":     "marios_birstonas_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "marios_birstonas_anfis_model_3d.pth",
            "scaler_x":   "marios_birstonas_scaler_X_3d.pkl",
            "scaler_y":   "marios_birstonas_scaler_y_3d.pkl",
            "config":     "marios_birstonas_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "marios_birstonas_anfis_model_5d.pth",
            "scaler_x":   "marios_birstonas_scaler_X_5d.pkl",
            "scaler_y":   "marios_birstonas_scaler_y_5d.pkl",
            "config":     "marios_birstonas_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
NEMUNAS_NEMAJUNAI_CONFIG = {
    "name": "nemunas_nemajunai",
    "display_name": "Nemunas (Nemajūnai)",
    "data_file": "live_data_nemunas_nemajunai.csv",
    "predictions_log_path": "predictions_log_nemunas_nemajunai.json",
    "hydro_station": "nemajunu-vms",
    "lat": 54.554227,
    "lon": 24.072006,
    "meteo_stations": ["birstono-ams", "alytaus-ams"],
    "risk_levels": [300, 500, 700],
    "horizons": {
        "1d": {
            "model":      "nemunas_nemajunai_anfis_model.pth",
            "scaler_x":   "nemunas_nemajunai_scaler_X.pkl",
            "scaler_y":   "nemunas_nemajunai_scaler_y.pkl",
            "config":     "nemunas_nemajunai_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "nemunas_nemajunai_anfis_model_3d.pth",
            "scaler_x":   "nemunas_nemajunai_scaler_X_3d.pkl",
            "scaler_y":   "nemunas_nemajunai_scaler_y_3d.pkl",
            "config":     "nemunas_nemajunai_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "nemunas_nemajunai_anfis_model_5d.pth",
            "scaler_x":   "nemunas_nemajunai_scaler_X_5d.pkl",
            "scaler_y":   "nemunas_nemajunai_scaler_y_5d.pkl",
            "config":     "nemunas_nemajunai_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
NEMUNAS_DARSUNISKIS_CONFIG = {
    "name": "nemunas_darsuniskis",
    "display_name": "Nemunas (Darsūniškis)",
    "data_file": "live_data_nemunas_darsuniskis.csv",
    "predictions_log_path": "predictions_log_nemunas_darsuniskis.json",
    "hydro_station": "darsuniskio-vms",
    "lat": 54.3700,
    "lon": 24.2100,
    "meteo_stations": ["kauno-ams"],
    "risk_levels": [300, 500, 700],
    "horizons": {
        "1d": {
            "model":      "nemunas_darsuniskis_anfis_model.pth",
            "scaler_x":   "nemunas_darsuniskis_scaler_X.pkl",
            "scaler_y":   "nemunas_darsuniskis_scaler_y.pkl",
            "config":     "nemunas_darsuniskis_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "nemunas_darsuniskis_anfis_model_3d.pth",
            "scaler_x":   "nemunas_darsuniskis_scaler_X_3d.pkl",
            "scaler_y":   "nemunas_darsuniskis_scaler_y_3d.pkl",
            "config":     "nemunas_darsuniskis_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "nemunas_darsuniskis_anfis_model_5d.pth",
            "scaler_x":   "nemunas_darsuniskis_scaler_X_5d.pkl",
            "scaler_y":   "nemunas_darsuniskis_scaler_y_5d.pkl",
            "config":     "nemunas_darsuniskis_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
NEMUNAS_LAMPEDZIAI_CONFIG = {
    "name": "nemunas_lampedziai",
    "display_name": "Nemunas (Lampėdžiai)",
    "data_file": "live_data_nemunas_lampedziai.csv",
    "predictions_log_path": "predictions_log_nemunas_lampedziai.json",
    "hydro_station": "lampedziu-vms",
    "lat": 54.906414,
    "lon": 23.817574,
    "meteo_stations": ["kauno-ams"],
    "risk_levels": [300, 500, 700],
    "horizons": {
        "1d": {
            "model":      "nemunas_lampedziai_anfis_model.pth",
            "scaler_x":   "nemunas_lampedziai_scaler_X.pkl",
            "scaler_y":   "nemunas_lampedziai_scaler_y.pkl",
            "config":     "nemunas_lampedziai_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "nemunas_lampedziai_anfis_model_3d.pth",
            "scaler_x":   "nemunas_lampedziai_scaler_X_3d.pkl",
            "scaler_y":   "nemunas_lampedziai_scaler_y_3d.pkl",
            "config":     "nemunas_lampedziai_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "nemunas_lampedziai_anfis_model_5d.pth",
            "scaler_x":   "nemunas_lampedziai_scaler_X_5d.pkl",
            "scaler_y":   "nemunas_lampedziai_scaler_y_5d.pkl",
            "config":     "nemunas_lampedziai_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
VILNIA_VILNIUS_CONFIG = {
    "name": "vilnia_vilnius",
    "display_name": "Vilnia (Vilnius)",
    "data_file": "live_data_vilnia_vilnius.csv",
    "predictions_log_path": "predictions_log_vilnia_vilnius.json",
    "hydro_station": "vilniaus-vilnia-vms",
    "lat": 54.6872,
    "lon": 25.2797,
    "meteo_stations": ["vilniaus-ams"],
    "risk_levels": [150, 280, 400],
    "horizons": {
        "1d": {
            "model":      "vilnia_vilnius_anfis_model.pth",
            "scaler_x":   "vilnia_vilnius_scaler_X.pkl",
            "scaler_y":   "vilnia_vilnius_scaler_y.pkl",
            "config":     "vilnia_vilnius_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "vilnia_vilnius_anfis_model_3d.pth",
            "scaler_x":   "vilnia_vilnius_scaler_X_3d.pkl",
            "scaler_y":   "vilnia_vilnius_scaler_y_3d.pkl",
            "config":     "vilnia_vilnius_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "vilnia_vilnius_anfis_model_5d.pth",
            "scaler_x":   "vilnia_vilnius_scaler_X_5d.pkl",
            "scaler_y":   "vilnia_vilnius_scaler_y_5d.pkl",
            "config":     "vilnia_vilnius_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}
NERIS_VILNIUS_CONFIG = {
    "name": "neris_vilnius",
    "display_name": "Neris (Vilnius)",
    "data_file": "live_data_neris_vilnius.csv",
    "predictions_log_path": "predictions_log_neris_vilnius.json",
    "hydro_station": "vilniaus-neris-vms",
    "lat": 54.6872,
    "lon": 25.2797,
    "meteo_stations": ["vilniaus-ams"],
    "risk_levels": [150, 280, 400],
    "horizons": {
        "1d": {
            "model":      "neris_vilnius_anfis_model.pth",
            "scaler_x":   "neris_vilnius_scaler_X.pkl",
            "scaler_y":   "neris_vilnius_scaler_y.pkl",
            "config":     "neris_vilnius_training_config.json",
            "label":      "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model":      "neris_vilnius_anfis_model_3d.pth",
            "scaler_x":   "neris_vilnius_scaler_X_3d.pkl",
            "scaler_y":   "neris_vilnius_scaler_y_3d.pkl",
            "config":     "neris_vilnius_training_config_3d.json",
            "label":      "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model":      "neris_vilnius_anfis_model_5d.pth",
            "scaler_x":   "neris_vilnius_scaler_X_5d.pkl",
            "scaler_y":   "neris_vilnius_scaler_y_5d.pkl",
            "config":     "neris_vilnius_training_config_5d.json",
            "label":      "5-Day Forecast",
            "days_ahead": 5,
        },
    },
}







RIVER_CONFIGS = {
    "minija":           MINIJA_CONFIG,
    "dane":             DANE_CONFIG,
    "kartena":          KARTENA_CONFIG,
    "dane_kretinga":    DANE_KRETINGA_CONFIG,
    "kanalas_lankupiu": KANALAS_LANKUPIU_CONFIG,
    "lankupiu_minija":  LANKUPIU_MINIJA_CONFIG,
    "marios_birstonas":  MARIOS_BIRSTONAS_CONFIG,
    "nemunas_nemajunai":  NEMUNAS_NEMAJUNAI_CONFIG,
    "nemunas_darsuniskis":  NEMUNAS_DARSUNISKIS_CONFIG,
    "nemunas_lampedziai":   NEMUNAS_LAMPEDZIAI_CONFIG,
    "vilnia_vilnius":      VILNIA_VILNIUS_CONFIG,
    "neris_vilnius":       NERIS_VILNIUS_CONFIG,
}

# ---------------------------------------------------------------------------
# MF LABELS
# ---------------------------------------------------------------------------
MF_LABELS = {
    "API_norm":   {0: "Low",         1: "Medium",      2: "Extreme",     3: "High",        4: "Very High"},
    "S_t":        {0: "Late Winter",  1: "Mid Spring",  2: "Late Jan",    3: "Mid Spring",  4: "Early Spring"},
    "SMI_t":      {0: "Extreme",     1: "Very High",   2: "Medium",      3: "Low",         4: "High"},
    "Pt":         {0: "High",        1: "Medium",      2: "Extreme",     3: "Very High",   4: "Low"},
    "delta_WL_t": {0: "Very High",   1: "High",        2: "Medium",      3: "Extreme",     4: "Low"},
}


# ---------------------------------------------------------------------------
# DATA FETCHING & FEATURES
# ---------------------------------------------------------------------------
def fetch_water_level_latest(station_code):
    for days_back in [0, 1, 2]:
        date_str = (datetime.now().date() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = (
            f"https://api.meteo.lt/v1/hydro-stations/{station_code}"
            f"/observations/measured/{date_str}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("observations", [])
                valid = [x for x in data if x.get("waterLevel") is not None]
                if valid:
                    valid.sort(key=lambda x: x.get("observationTimeUtc", ""))
                    return round(valid[-1]["waterLevel"], 2)
        except Exception:
            pass
    return None


def fetch_station_forecast(station_code, main_forecast_level):
    try:
        current = fetch_water_level_latest(station_code)
        if current is None:
            variation = np.random.uniform(-0.1, 0.1)
            current = round(main_forecast_level * (1 + variation), 2)
        forecast = round(current * 1.05, 2)
        return current, forecast
    except Exception as e:
        logger.warning(f"Failed to fetch data for {station_code}: {e}")
        return round(main_forecast_level * 0.95, 2), round(main_forecast_level * 1.02, 2)


def fetch_meteo_latest(station_code):
    date_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.meteo.lt/v1/stations/{station_code}/observations/{date_str}"
    try:
        resp = requests.get(url, timeout=10)
        obs = resp.json().get("observations", [])
        precip = sum(o.get("precipitation", 0) for o in obs if o.get("precipitation") is not None)
        temps  = [o["airTemperature"] for o in obs if o.get("airTemperature") is not None]
        avg_t  = sum(temps) / len(temps) if temps else 0
        return round(precip, 2), round(avg_t, 2)
    except Exception:
        return 0, 0


def prepare_complex_features(df):
    df = df.copy().sort_values("timestamp")
    p_cols = [c for c in df.columns if "precip" in c]
    df["Pt"] = df[p_cols].mean(axis=1) if p_cols else 0

    api_vals, current_api = [], 0
    for p in df["Pt"]:
        current_api = p + (K_DECAY * current_api)
        api_vals.append(current_api)
    df["API_t"] = api_vals

    a_min, a_max = df["API_t"].min(), df["API_t"].max()
    df["API_norm"] = (df["API_t"] - a_min) / (a_max - a_min) if a_max != a_min else 0

    d = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["S_t"] = np.cos((2 * np.pi * d) / 365)

    t_cols = [c for c in df.columns if "temp" in c]
    if t_cols:
        avg_temp = df[t_cols].mean(axis=1)
        df["SMI_t"] = avg_temp.apply(lambda x: max(0, x * 2.5) if x > 0 else 0)
    else:
        df["SMI_t"] = 0

    df["delta_WL_t"] = df["water_level_cm"].diff().fillna(0)
    return df


def extract_strongest_rule(model, X_scaled, num_mfs):
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float()
        fuzzified = model.layer["fuzzify"](X_tensor)
        firing_strengths = model.layer["rules"](fuzzified)
    strengths  = firing_strengths[0].numpy()
    rule_id    = int(np.argmax(strengths))
    activation = float(strengths[rule_id])
    mf_indices = []
    temp = rule_id
    for _ in range(5):
        mf_indices.append(temp % num_mfs)
        temp //= num_mfs
    return rule_id, activation, list(reversed(mf_indices))


# ---------------------------------------------------------------------------
# PREDICTION JOB
# ---------------------------------------------------------------------------
def run_prediction_job(config):
    logger.info(f"--- Running Prediction Cycle for {config['display_name']} ---")
    try:
        live_wl = fetch_water_level_latest(config["hydro_station"])
        if live_wl is None:
            logger.warning(
                f"[{config['display_name']}] No water level returned "
                f"for station '{config['hydro_station']}'. Skipping."
            )
            return

        logger.info(f"[{config['display_name']}] Live water level: {live_wl} cm")

        today_str = datetime.now().strftime("%Y-%m-%d")
        live_row  = {"timestamp": today_str, "water_level_cm": live_wl}
        for s in config["meteo_stations"]:
            p, t = fetch_meteo_latest(s)
            live_row[f"precip_{s}_mm"] = p
            live_row[f"temp_{s}_c"]    = t

        df_hist = (
            pd.read_csv(config["data_file"])
            if os.path.exists(config["data_file"])
            else pd.DataFrame()
        )
        df_comb = (
            pd.concat([df_hist, pd.DataFrame([live_row])], ignore_index=True)
            .drop_duplicates("timestamp")
        )
        df_comb.to_csv(config["data_file"], index=False)
        df_feats = prepare_complex_features(df_comb)

        last_row = df_feats.iloc[-1]
        feature_display = {
            "Water Level":           f"{live_wl} cm",
            "Precipitation (Pt)":    f"{round(last_row['Pt'], 2)} mm",
            "Soil Saturation (API)": f"{round(last_row['API_t'], 2)} idx",
            "Snowmelt (SMI)":        f"{round(last_row['SMI_t'], 2)} mm",
            "Seasonality":           f"{round(last_row['S_t'], 3)}",
            "Trend":                 f"{round(last_row['delta_WL_t'], 2)} cm",
        }

        all_preds     = {}
        dominant_rule = ""
        input_data    = last_row[["API_norm", "S_t", "SMI_t", "Pt", "delta_WL_t"]].values.reshape(1, -1)

        for h_id, h_cfg in config["horizons"].items():

            missing = [
                f for f in [h_cfg["model"], h_cfg["scaler_x"], h_cfg["scaler_y"], h_cfg["config"]]
                if not os.path.exists(f)
            ]
            if missing:
                logger.warning(
                    f"[{config['display_name']}] {h_id}: skipping — missing files: {missing}"
                )
                continue

            scaler_x = joblib.load(h_cfg["scaler_x"])
            scaler_y = joblib.load(h_cfg["scaler_y"])

            with open(h_cfg["config"], "r") as f:
                num_mfs = json.load(f).get("num_mfs", 5)

            X_scaled = scaler_x.transform(input_data)

            centres   = torch.linspace(0.1, 0.9, num_mfs)
            invardefs = [
                (
                    f"x{i}",
                    [
                        BellMembFunc(
                            torch.tensor([1.0]),
                            torch.tensor([2.0]),
                            centres[j].unsqueeze(0),
                        )
                        for j in range(num_mfs)
                    ]
                )
                for i in range(5)
            ]
            model = AnfisNet(f"ANFIS_{h_id}", invardefs, ["y"], hybrid=True)
            ckpt  = torch.load(h_cfg["model"], map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])

            coeff = ckpt.get("coeff")
            if coeff is None:
                coeff = ckpt.get("consequent_coeffs")
            if coeff is None:
                logger.error(
                    f"[{config['display_name']}] {h_id}: no coeff in checkpoint — skipping."
                )
                continue
            if isinstance(coeff, np.ndarray):
                coeff = torch.tensor(coeff, dtype=torch.float32)
            model.coeff = coeff
            model.eval()

            with torch.no_grad():
                raw = model(torch.from_numpy(X_scaled).float())

            if torch.isnan(raw).any():
                logger.error(
                    f"[{config['display_name']}] {h_id}: model output NaN — skipping."
                )
                continue

            pred_chg = float(scaler_y.inverse_transform(raw.numpy())[0, 0])

            all_preds[h_id] = {
                "level":  float(round(live_wl + pred_chg, 2)),
                "change": float(round(pred_chg, 2)),
                "label":  h_cfg["label"],
            }

            if h_id == "1d":
                r_id, act, mf_i = extract_strongest_rule(model, X_scaled, num_mfs)
                parts = [
                    f"{fn} is {MF_LABELS[fn][mf_i[i]]}"
                    for i, fn in enumerate(["API_norm", "S_t", "SMI_t", "Pt", "delta_WL_t"])
                ]
                dominant_rule = f"Rule #{r_id} (Act: {act:.2f}): IF " + " AND ".join(parts)

        if not all_preds:
            logger.warning(f"[{config['display_name']}] No valid predictions produced.")
            return

        log_path = config["predictions_log_path"]
        tmp_path = log_path + ".tmp"
        log_data = {}
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted log detected. Resetting.")

        log_data[today_str] = {
            "actual":     float(live_wl),
            "features":   feature_display,
            "fired_rule": dominant_rule,
            "horizons":   all_preds,
        }
        with open(tmp_path, "w") as f:
            json.dump(log_data, f, indent=4)
        os.replace(tmp_path, log_path)
        logger.info(f"[{config['display_name']}] Log written for {today_str}")

    except Exception as e:
        logger.error(f"Job Failed for {config['display_name']}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/rivers")
def get_rivers():
    return jsonify([
        {
            "id":           key,
            "display_name": cfg["display_name"],
            "lat":          cfg["lat"],
            "lon":          cfg["lon"],
        }
        for key, cfg in RIVER_CONFIGS.items()
    ])


@app.route("/api/data")
def get_data_api():
    river  = request.args.get("river", "minija")
    config = RIVER_CONFIGS.get(river)
    if not config:
        return jsonify({"error": "River not found"}), 404

    path = config["predictions_log_path"]
    if not os.path.exists(path):
        return jsonify({"error": "No prediction data yet for this river."}), 404

    try:
        with open(path, "r") as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Corrupted log"}), 500

    sorted_dates = sorted(log_data.keys())
    last_date    = sorted_dates[-1]
    last_entry   = log_data[last_date]

    chart_data = []
    for d in sorted_dates:
        entry     = log_data[d]
        base_date = datetime.strptime(d, "%Y-%m-%d")

        chart_data.append({
            "date":   d,
            "actual": entry.get("actual"),
            "pred1d": None,
            "pred3d": None,
            "pred5d": None,
        })

        horizons = entry.get("horizons", {})
        for h_id, days, key in [("1d", 1, "pred1d"), ("3d", 3, "pred3d"), ("5d", 5, "pred5d")]:
            if horizons.get(h_id):
                row = {k: None for k in ["date", "actual", "pred1d", "pred3d", "pred5d"]}
                row["date"] = (base_date + timedelta(days=days)).strftime("%Y-%m-%d")
                row[key]    = horizons[h_id]["level"]
                chart_data.append(row)

    main_forecast = last_entry.get("horizons", {}).get("1d", {}).get("level", 200)
    stations_data = []
    for station in MONITORING_STATIONS.get(river, []):
        if station["is_main"]:
            current  = last_entry.get("actual")
            forecast = main_forecast
        else:
            current, forecast = fetch_station_forecast(station["code"], main_forecast)
        stations_data.append({
            "name":     station["name"],
            "lat":      station["lat"],
            "lon":      station["lon"],
            "current":  current,
            "forecast": forecast,
        })

    return jsonify({
        "riverName":       config["display_name"],
        "riverKey":        river,
        "lat":             config["lat"],
        "lon":             config["lon"],
        "riskLevels":      config["risk_levels"],
        "lastKnownLevel":  last_entry.get("actual"),
        "horizons":        last_entry.get("horizons"),
        "currentFeatures": last_entry.get("features"),
        "firedRule":       last_entry.get("fired_rule"),
        "historicalData":  chart_data[-30:],
        "lastUpdated":     datetime.now().isoformat(),
        "stations":        stations_data,
        "infrastructure":  INFRASTRUCTURE_DATA.get(river, []),
    })
@app.route("/api/summary")
def get_ai_summary():
    try:
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
                h      = last_entry.get("horizons", {})
                h1d    = h.get("1d", {}).get("level", "—")
                h3d    = h.get("3d", {}).get("level", "—")
                h5d    = h.get("5d", {}).get("level", "—")
                risk_levels = cfg["risk_levels"]
                level_val   = h1d if isinstance(h1d, (int, float)) else actual

                if isinstance(level_val, (int, float)):
                    r1, r2, r3 = risk_levels
                    if level_val > r3:
                        risk = "SEVERE"
                    elif level_val > r2:
                        risk = "HIGH"
                    elif level_val > r1:
                        risk = "MODERATE"
                    else:
                        risk = "LOW"
                else:
                    risk = "UNKNOWN"

                river_lines.append(
                    f"- {cfg['display_name']}: current={actual} cm, "
                    f"1d={h1d} cm, 3d={h3d} cm, 5d={h5d} cm, risk={risk}"
                )
            except Exception:
                continue

        if not river_lines:
            return jsonify({"summary": "No river data available yet."}), 200

        prompt = (
            "You are a flood risk analyst. Based on the following real-time water level "
            "readings and forecasts for Lithuanian rivers, write a concise 3-4 sentence "
            "summary for emergency management personnel. Highlight which rivers pose the "
            "highest immediate risk, any concerning trends, and which are currently safe. "
            "Use plain language, no bullet points, no markdown.\n\n"
            "River data:\n" + "\n".join(river_lines)
        )

        # ── New google.genai API call ──────────────────────────────────
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = response.text.strip()

        return jsonify({
            "summary":     summary,
            "river_count": len(river_lines),
            "generated_at": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Gemini summary failed: {e}", exc_info=True)
        return jsonify({"summary": "AI summary unavailable.", "error": str(e)}), 500




# ---------------------------------------------------------------------------
# SCHEDULER
# ---------------------------------------------------------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(func=lambda: run_prediction_job(MINIJA_CONFIG),            trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(DANE_CONFIG),              trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(KARTENA_CONFIG),           trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(DANE_KRETINGA_CONFIG),     trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(KANALAS_LANKUPIU_CONFIG),  trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(LANKUPIU_MINIJA_CONFIG),   trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(MARIOS_BIRSTONAS_CONFIG),  trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(NEMUNAS_NEMAJUNAI_CONFIG), trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(NEMUNAS_DARSUNISKIS_CONFIG), trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(NEMUNAS_LAMPEDZIAI_CONFIG), trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(VILNIA_VILNIUS_CONFIG), trigger="cron", minute="05")
scheduler.add_job(func=lambda: run_prediction_job(NERIS_VILNIUS_CONFIG), trigger="cron", minute="05")  # ← NEW
scheduler.start()


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
def run_startup_jobs():
    import threading

    def _run_one(cfg):
        try:
            run_prediction_job(cfg)
        except Exception as e:
            logger.error(f"Startup job failed for {cfg['display_name']}: {e}")

    def _startup():
        logger.info("=== Running startup prediction jobs ===")
        time.sleep(2)

        configs = [
            MINIJA_CONFIG,
            DANE_CONFIG,
            KARTENA_CONFIG,
            DANE_KRETINGA_CONFIG,
            KANALAS_LANKUPIU_CONFIG,
            LANKUPIU_MINIJA_CONFIG,
            MARIOS_BIRSTONAS_CONFIG,
            NEMUNAS_NEMAJUNAI_CONFIG,
            NEMUNAS_DARSUNISKIS_CONFIG,
            NEMUNAS_LAMPEDZIAI_CONFIG,
            VILNIA_VILNIUS_CONFIG,
            NERIS_VILNIUS_CONFIG,
        ]

        threads = [
            threading.Thread(target=_run_one, args=(cfg,), daemon=True)
            for cfg in configs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.info("=== Startup prediction jobs complete ===")

    threading.Thread(target=_startup, daemon=True).start()

run_startup_jobs()

if __name__ == "__main__":
    app.run(debug=True, port=5002, use_reloader=False)



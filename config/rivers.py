"""
River configuration definitions.
Each config defines the river name, data paths, hydro/meteo stations,
risk thresholds, and model files for each forecast horizon.
"""

import os

# Base directories for models, scalers, and training configs
_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(_ROOT_DIR, "ANFIS_models")
SCALERS_DIR = os.path.join(_ROOT_DIR, "Scalers")
CONFIGS_DIR = os.path.join(_ROOT_DIR, "Training_configs")


def _horizons(prefix, days_list=(1, 3, 5)):
    """Generate horizon configs from a prefix and list of day offsets."""
    suffix_map = {1: "1d", 3: "3d", 5: "5d"}
    label_map = {1: "1-Day Forecast", 3: "3-Day Forecast", 5: "5-Day Forecast"}

    horizons = {}
    for d in days_list:
        tag = suffix_map[d]
        file_suffix = f"_{tag}" if d > 1 else ""
        horizons[tag] = {
            "model": os.path.join(MODELS_DIR, f"{prefix}_anfis_model{file_suffix}.pth"),
            "scaler_x": os.path.join(SCALERS_DIR, f"{prefix}_scaler_X{file_suffix}.pkl"),
            "scaler_y": os.path.join(SCALERS_DIR, f"{prefix}_scaler_y{file_suffix}.pkl"),
            "config": os.path.join(CONFIGS_DIR, f"{prefix}_training_config{file_suffix}.json"),
            "label": label_map[d],
            "days_ahead": d,
        }
    return horizons


# ---------------------------------------------------------------------------
# Individual river configs
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
            "model": os.path.join(MODELS_DIR, "anfis_model.pth"),
            "scaler_x": os.path.join(SCALERS_DIR, "scaler_X.pkl"),
            "scaler_y": os.path.join(SCALERS_DIR, "scaler_Y.pkl"),
            "config": os.path.join(CONFIGS_DIR, "training_config.json"),
            "label": "1-Day Forecast",
            "days_ahead": 1,
        },
        "3d": {
            "model": os.path.join(MODELS_DIR, "anfis_model_3d.pth"),
            "scaler_x": os.path.join(SCALERS_DIR, "scaler_X_3d.pkl"),
            "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_3d.pkl"),
            "config": os.path.join(CONFIGS_DIR, "training_config_3d.json"),
            "label": "3-Day Forecast",
            "days_ahead": 3,
        },
        "5d": {
            "model": os.path.join(MODELS_DIR, "anfis_model_5d.pth"),
            "scaler_x": os.path.join(SCALERS_DIR, "scaler_X_5d.pkl"),
            "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_5d.pkl"),
            "config": os.path.join(CONFIGS_DIR, "training_config_5d.json"),
            "label": "5-Day Forecast",
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
    "horizons": _horizons("dane"),
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
    "horizons": _horizons("kartena"),
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
    "horizons": _horizons("dane_kretinga"),
}

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
    "horizons": _horizons("kanalas_lankupiu"),
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
    "horizons": _horizons("lankupiu_minija"),
}

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
    "horizons": _horizons("marios_birstonas"),
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
    "horizons": _horizons("nemunas_nemajunai"),
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
    "horizons": _horizons("nemunas_darsuniskis"),
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
    "horizons": _horizons("nemunas_lampedziai"),
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
    "horizons": _horizons("vilnia_vilnius"),
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
    "horizons": _horizons("neris_vilnius"),
}


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------
RIVER_CONFIGS = {
    "minija": MINIJA_CONFIG,
    "dane": DANE_CONFIG,
    "kartena": KARTENA_CONFIG,
    "dane_kretinga": DANE_KRETINGA_CONFIG,
    "kanalas_lankupiu": KANALAS_LANKUPIU_CONFIG,
    "lankupiu_minija": LANKUPIU_MINIJA_CONFIG,
    "marios_birstonas": MARIOS_BIRSTONAS_CONFIG,
    "nemunas_nemajunai": NEMUNAS_NEMAJUNAI_CONFIG,
    "nemunas_darsuniskis": NEMUNAS_DARSUNISKIS_CONFIG,
    "nemunas_lampedziai": NEMUNAS_LAMPEDZIAI_CONFIG,
    "vilnia_vilnius": VILNIA_VILNIUS_CONFIG,
    "neris_vilnius": NERIS_VILNIUS_CONFIG,
}


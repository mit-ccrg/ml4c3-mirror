PAIRWISE_DISTANCES = {
    "cityblock": [],
    "cosine": [],
    "euclidean": [],
    "l1": [],
    "l2": [],
    "manhattan": [],
    "braycurtis": [],
    "canberra": [],
    "chebyshev": [],
    "correlation": [],
    "dice": [],
    "hamming": [],
    "jaccard": [],
    "kulsinski": [],
    "mahalanobis": [],
    "minkowski": ["p"],
    "rogerstanimoto": [],
    "russellrao": [],
    "seuclidean": [],
    "sokalmichener": [],
    "sokalsneath": [],
    "sqeuclidean": [],
    "yule": [],
}

SIGNAL_PATHS = {
    "bedmaster": {
        "vitals": ["hr", "pa2s", "pa2d", "pa2m", "art1s", "art1d", "art1m"],
        "waveform": ["pa2", "art1", "spo2"],
    },
    "edw": {
        "flowsheet": [],
        "labs": [],
        "med": ["norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh"],
        "surgery": [],
        "transfusions": [],
    },
}

METADATA_MEDS = {
    "norepinephrine (mcg/ml)": "norepinephrine_infusion_syringe_in_swfi_80_mcg|ml_cmpd_central_mgh",
}

SIGNAL_LIMITS = {
    "hr": (20, 200),
    "pa2s": (0, 100),
    "pa2m": (0, 100),
    "pa2d": (0, 100),
    "art1s": (30, 220),
    "art1m": (30, 220),
    "art1d": (30, 220),
}

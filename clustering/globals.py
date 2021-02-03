# pylint: disable=line-too-long

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

DEPARTMENT_NAMES = {
    "BLK08": "MGH BLAKE 8 CARD SICU",
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
    "bt": (2.0, 5.0),
    "spo2r": (0, 300),
}

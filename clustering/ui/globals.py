# Imports: first party
from clustering.main import (
    cluster,
    distances,
    curate_data,
    in_pad_data,
    outpad_data,
    normalize_data,
    downsample_data,
)
from clustering.objects.modifiers import (
    Padder,
    Cluster,
    Normalizer,
    Downsampler,
    DistanceMetric,
    OutlierRemover,
    DimensionsReducer,
)

TITLE = "Clustering process"
STEPS = {
    "preprocess": {
        "outlier_removal": OutlierRemover,
        "in-padding": Padder,
        "out-padding": Padder,
        "down-sampling": Downsampler,
        "normalization": Normalizer,
    },
    "distance": {"distance": DistanceMetric},
    "cluster": {
        "cluster": Cluster,
        "cluster-distance": DistanceMetric,
        "cluster-algo": DistanceMetric,
    },
    "reduce": {"reduce": DimensionsReducer},
}
PROCESSES = {
    "outlier_removal": curate_data,
    "in-padding": in_pad_data,
    "out-padding": outpad_data,
    "down-sampling": downsample_data,
    "normalization": normalize_data,
    "distance": distances,
    "cluster": cluster,
}

# Imports: third party
from hyperoptimize import (
    _test_build_downstream_model,
    _test_build_pretraining_model,
    _test_build_downstream_model_pretrained,
    _test_build_pretraining_model_bad_group_size,
)

print(100 * "~")
_test_build_pretraining_model()
print(100 * "~")
_test_build_pretraining_model_bad_group_size()
print(100 * "~")
_test_build_downstream_model()
print(100 * "~")
_test_build_downstream_model_pretrained()

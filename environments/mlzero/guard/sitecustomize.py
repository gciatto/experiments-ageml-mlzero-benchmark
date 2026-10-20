"""Inference-time fit detector, loaded through PYTHONPATH in the inference container.

Records (does not block) calls to fitting entry points of common ML libraries
into $AGEML_BENCH_FIT_LOG. A non-empty log is a no-fitting contract violation
for the benchmark, pending manual review of the recorded call sites.
"""

import json
import os
import sys
import threading

LOG = os.environ.get("AGEML_BENCH_FIT_LOG")
NAMES = {"fit", "partial_fit", "fit_transform", "fit_predict", "refit_full", "train"}
LIBS = ("sklearn", "autogluon", "lightgbm", "xgboost", "catboost", "imblearn")
_seen = set()


def _profile(frame, event, arg):
    if event != "call":
        return
    code = frame.f_code
    if code.co_name not in NAMES:
        return
    path = code.co_filename.replace(os.sep, "/")
    if "/site-packages/" not in path or not any(f"/{lib}/" in path for lib in LIBS):
        return
    site = f"{path}:{code.co_firstlineno}:{code.co_name}"
    if site in _seen:
        return
    _seen.add(site)
    with open(LOG, "a") as f:
        f.write(json.dumps({"call": site}) + "\n")


if LOG:
    # ponytail: setprofile slows pure-Python inference; sys.monitoring once on 3.12+
    sys.setprofile(_profile)
    threading.setprofile(_profile)

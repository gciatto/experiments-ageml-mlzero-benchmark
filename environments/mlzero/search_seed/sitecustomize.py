"""Seed newly launched Python processes; no guarantee of backend determinism."""

import os
import random

seed = int(os.environ.get("AGEML_SEARCH_SEED", "42"))
random.seed(seed)
try:
    import numpy

    numpy.random.seed(seed)
except ImportError:
    pass

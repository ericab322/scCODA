import time
import tracemalloc
import os
import pandas as pd

import numpy as np
import pandas as pd


def profile_time_memory(func, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()

    result = func(*args, **kwargs)

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profile = {
        "elapsed_time_seconds": end - start,
        "current_memory_MB": current / 1e6,
        "peak_memory_MB": peak / 1e6,
    }
    return result, profile


def log_profile_row(out_csv, metadata, profile):
    row = {**metadata, **profile}
    df = pd.DataFrame([row])

    # append if file exists, otherwise write with header
    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)

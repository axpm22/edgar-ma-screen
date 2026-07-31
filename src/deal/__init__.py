"""Deal-prediction package.

OpenMP is pinned before LightGBM loads. Left unset, libomp opens a
thread pool per core; with several training processes running that
oversubscribes the CPU and macOS emits "count=0, state=0xa" contention
warnings. Two threads per process keeps the machine usable.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

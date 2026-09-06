"""Run the pinned timing bench against whatever engine revision is checked out."""
import sys, pathlib
tests = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tests))
import test_perf_ratchet as T
T.test_perf_timing_bench()
print(f'SKIPPED={len(T.SKIPPED)}')

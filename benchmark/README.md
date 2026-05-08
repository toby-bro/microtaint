# MicroTaint test artifacts

In this sub directory are all the test files run and used to run the benchmarks mentionned in the MicroTaint submission.

## General comparison benchmark

You need to go into the precision_soundess directory to run these commands

The overall benchmark needs a few tools to work, the script that sets them all up is `setup-envs.sh`

In order to install all the different tools in their correct version, a working [uv](https://docs.astral.sh/uv), git and docker will be needed.

Once both work you can run
```sh
./setup-envs.sh
```
It will build the docker images, download all the missing files and install the dependencies in the local dir in which you are in.

Once this is done you can run the benchmark we ran in the submission with the following command

```sh
BATCH_TIMEOUT=18000 uv run benchmark.py  --number 7500   --sequences 1000  --sweep  --all-suites  --quiet  --seed 12 -w taintgrind,libdft64,microtaint,triton,maat,angr,panda 2>&1 | tee benchmark_$(date +%Y%m%d_%H%M%S).log
```

The `BATCH_TIMEOUT` can be adjusted 18000 corresponds to the number of seconds in 5 hours. 

The slow engines amongst these are panda,maat, libdft64, taintgrind

You can adjust the engines used through the `-w` flag, to change the number of bits used in the ground truth threshold then you can adjust the variable `GT_BIT_BUDGET` on line 218.

The test drops 25 test cases, for which we did not resolve a bug at the time of submission in the harness, we exclude 4 cases that broke the ground truth oracle, that we had not time enough to check at the time of submission.

The rep based tests are now fixed.

## Overhead

This set of tests is in the overhead directory
The command to run is 

```sh
uv run overhead_bench.py --build-bench bench.c --gen-input 256 --runs 100 --only native --only qiling-only --only microtaint-all --native-timeout 5 --qiling-timeout 120 --microtaint-timeout 1800 --json overhead_results.json
```


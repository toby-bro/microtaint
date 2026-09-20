# Reference run

One run of `./run-all.sh --quick --no-baselines` on a minimal Debian 12 machine.
It is in `run-all-debian12/20260918-121340`, with the summary, the build log, and the stdout, the stderr and the JSON output of every experiment.

The results agree with the paper.
The avalanche figures are identical (base64 370 instructions, 16.1% of data bits and 53.3% of flag bits; nftables 35 and 60.0%; siphash 325 and 63.6%), and the five architectures each compared 20000 cases with no under-taint.

The overhead figures are the exception.
The machine is virtualised, so everything is slower and the timings are noisier than on real hardware: the emulation floor is 0.588 s here against 0.385 s in the paper, and the rest of the ladder follows.
The ordering of the rungs is what to check, not the exact magnitudes.

The engine comparison is absent because `--no-baselines` skips it, and with it the two macro generation steps at the end.

## Running it

The machine was a Debian 12 `genericcloud` image with eight cores and 4 GB of memory, holding 324 packages, no compiler and no git.

```sh
sudo apt-get update
sudo apt-get install -y git build-essential      # curl is already in the image

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone <the artifact repository> microtaint
cd microtaint
git checkout v0.7.2
uv sync --locked --all-extras

cd artifacts/ndss27
./run-all.sh --quick --no-baselines
```

It took one hour and forty five minutes.

## Running everything

The six compared engines and TaintInduce need more.
Docker must work without `sudo`, and group membership only takes effect in a new login session, so log out and back in after the `usermod`.

```sh
sudo apt-get install -y docker.io valgrind wget openssl
sudo usermod -aG docker "$USER"
# log out and back in here

cd microtaint/artifacts/ndss27
./rq2-comparison/setup_envs.sh                   # the six engines, about 5 GB
./rq1-synthesis_vs_inference/setup_taintinduce.sh

./run-all.sh
```

There is also a `./setup-all.sh` that runs both setup scripts and checks the dependencies first.
It is what we used on a fresh Debian 12 and it is only tested there, so on any other distribution the commands above are the reference and the package names are yours to translate.

`valgrind` is needed on the host and not only in the container, because the TaintGrind harness is compiled here against `/usr/include/valgrind`.
Both setup scripts can be run again if the network drops part way through; they pick up where they stopped rather than starting over.
`setup_envs.sh` downloads about 5 GB, most of it PANDA's guest image, and the whole tree then occupies 15 GB.
The full `./run-all.sh` takes about two days, dominated by the cross-ISA campaign.

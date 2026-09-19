# ---------------------------------------------------------------------------
# Dependency preflight
#
# Every missing dependency used to announce itself in its own way, hours apart
# and deep inside an experiment: `cc` not found from uv's build, `valgrind.h`
# missing from a C harness compile, docker refusing a socket.  Check them all
# here and say what to install, once, before anything runs.
# ---------------------------------------------------------------------------
check_deps() {  # check_deps <1 if the baselines are wanted, else 0>
  local want_baselines=${1:-0}
  local missing=() hints=()

  # Core: needed by every experiment, baselines or not.
  #   gcc/make  microtaint's C+Cython extensions, and the guest binaries
  #   python3   used by this script and by the harnesses
  #   tar/curl  fetching and unpacking the pinned base64 workloads
  #   ar        extracting base64 from the pinned .deb (binutils)
  for c in gcc make python3 tar curl ar; do
    command -v "$c" >/dev/null 2>&1 || missing+=("$c")
  done
  [ ${#missing[@]} -gt 0 ] && hints+=("  sudo apt install build-essential curl tar binutils")

  if [ "$want_baselines" -eq 1 ]; then
    local bmissing=()
    for c in docker wget openssl git; do
      command -v "$c" >/dev/null 2>&1 || bmissing+=("$c")
    done
    # benchmark.py compiles TaintGrind's C harness on the HOST against these
    # headers; without them that engine drops out of the comparison.
    [ -f /usr/include/valgrind/valgrind.h ] || bmissing+=("valgrind headers (/usr/include/valgrind/valgrind.h)")
    if [ ${#bmissing[@]} -gt 0 ]; then
      missing+=("${bmissing[@]}")
      hints+=("  sudo apt install docker.io valgrind wget openssl git   # for the baselines")
    fi
    # Membership in the docker group only takes effect in a NEW login session,
    # so being in the group is not the same as being able to use the socket.
    if command -v docker >/dev/null 2>&1 && ! docker info >/dev/null 2>&1; then
      missing+=("a usable docker socket (docker info failed)")
      hints+=("  sudo usermod -aG docker \"$USER\"   # then start a NEW login session")
    fi
  fi

  if [ ${#missing[@]} -gt 0 ]; then
    {
      echo "[!] missing ${#missing[@]} dependenc(y/ies):"
      printf '      - %s\n' "${missing[@]}"
      echo '    install:'
      printf '%s\n' "${hints[@]}"
      [ "$want_baselines" -eq 1 ] && echo '    or re-run with --no-baselines to skip RQ1 and RQ2.'
    } >&2
    return 1
  fi
  return 0
}

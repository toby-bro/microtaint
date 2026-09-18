# Locate uv, even when it is installed but not on PATH.  Source, do not execute:
#
#     . "$(dirname "$0")/../find_uv.sh"        # adjust the depth per script
#
# uv's own installer drops the binary in ~/.local/bin and then prints "restart
# your shell or run source $HOME/.local/bin/env".  Anyone who installs uv and
# runs one of these scripts in the SAME shell, or drives them non-interactively
# over ssh (where ~/.profile is never read), has a perfectly good uv that
# `command -v uv` cannot see.  The scripts then aborted telling the reader to go
# install the thing they had already installed.
#
# So look on PATH first, then in the places uv is normally installed, and put
# the one we find on PATH for every later call.
find_uv() {
    command -v uv >/dev/null 2>&1 && return 0

    local candidate
    for candidate in \
        "${UV_INSTALL_DIR:-}/uv" \
        "${XDG_BIN_HOME:-}/uv" \
        "${CARGO_HOME:-$HOME/.cargo}/bin/uv" \
        "$HOME/.local/bin/uv" \
        "$HOME/.cargo/bin/uv" \
        /usr/local/bin/uv \
        /snap/bin/uv \
        /opt/homebrew/bin/uv \
        /home/linuxbrew/.linuxbrew/bin/uv
    do
        case "$candidate" in /uv) continue ;; esac   # an unset variable above
        if [ -x "$candidate" ]; then
            PATH="$(dirname "$candidate"):$PATH"
            export PATH
            echo "[=] found uv at $candidate (added its directory to PATH)"
            return 0
        fi
    done

    echo '[!] uv not found, aborting... check https://docs.astral.sh/uv/ for installation instructions.' >&2
    echo '    If uv IS installed, it is not on PATH and not in a standard location;' >&2
    echo '    run `source $HOME/.local/bin/env` or add its directory to PATH.' >&2
    return 1
}

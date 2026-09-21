# Make the source tree buildable when it arrived as a tarball rather than a
# clone.  Source, do not execute.
#
# The package takes its version from git through hatch-vcs, so a tree with no
# .git cannot be built at all:
#
#     No distribution name is known here, so the per-distribution
#     SETUPTOOLS_SCM_PRETEND_VERSION_FOR_<DIST> form cannot match
#
# An archived artifact is exactly that tree.  Create a repository around it and
# commit once, so the version can be derived and so every experiment can record
# which tree produced its results.
#
# $ARTIFACT_VERSION overrides the tag, for a snapshot of some other release.
ensure_git() {
    local repo=$1
    [ -d "$repo/.git" ] && return 0

    command -v git >/dev/null 2>&1 || {
        echo '[!] this tree has no .git and git is not installed, so the version' >&2
        echo '    cannot be derived and the package cannot be built.' >&2
        return 1
    }

    echo '[*] no git repository here (an unpacked archive), creating one so the'
    echo '    version can be derived and the runs can record their provenance'
    # An identity is supplied inline so the commit does not depend on the
    # user having configured one, and signing is disabled explicitly: a global
    # commit.gpgsign=true otherwise makes this hang on a pinentry prompt that
    # nobody is there to answer, and the install fails with
    # "gpg: signing failed: Timeout".
    git -C "$repo" init -q
    git -C "$repo" add -A
    git -C "$repo" \
        -c user.name='artifact' -c user.email='artifact@localhost' \
        -c commit.gpgsign=false -c tag.gpgSign=false \
        commit -q -m 'artifacts snapshot' || return 1
    git -C "$repo" -c tag.gpgSign=false tag "${ARTIFACT_VERSION:-v0.7.3}"
    echo "[+] created a repository at $repo, tagged ${ARTIFACT_VERSION:-v0.7.3}"
}

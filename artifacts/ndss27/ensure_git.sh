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
# This is the one place a version still has to be stated, because an unpacked
# archive has no history to derive one from and every experiment records which
# tree produced it.  Set $ARTIFACT_VERSION when archiving a different release;
# the default below is only a fallback and will go stale.  Nothing else in the
# artifact pins a version: the repository is the living version and the Zenodo
# archive is the frozen one.
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
    git -C "$repo" -c tag.gpgSign=false tag "${ARTIFACT_VERSION:-v0.7.4}"
    echo "[+] created a repository at $repo, tagged ${ARTIFACT_VERSION:-v0.7.4}"
}

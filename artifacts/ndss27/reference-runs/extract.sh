#!/usr/bin/env sh
# Unpack the reference runs' archives in place.
#
# The bulky evidence is stored as .tar.xz because the engine comparison report
# is 12.7 MB of JSON and the synthesised rules are another 6.2 MB, which xz
# takes to 1.2 MB and 0.3 MB.  The scripts that read them want plain files, so
# unpack before running any of the commands in README.md:
#
#   ./extract.sh
#
# It is idempotent: an archive whose contents are already there is skipped, so
# it can be run again after a partial extraction.  Nothing is deleted, so the
# archives stay and a second run costs nothing.
set -eu
here=$(cd "$(dirname "$0")" && pwd)

for archive in "$here"/*/*.tar.xz; do
    [ -e "$archive" ] || continue
    dir=$(dirname "$archive")
    # The first entry tells us what this archive would create, which is what
    # makes the skip a check rather than a guess.
    first=$(tar tJf "$archive" | head -1)
    if [ -e "$dir/${first%%/*}" ]; then
        printf '  already there: %s\n' "${archive#"$here"/}"
        continue
    fi
    printf '  extracting:    %s\n' "${archive#"$here"/}"
    tar xJf "$archive" -C "$dir"
done
echo 'done'

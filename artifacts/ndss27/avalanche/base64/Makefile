# The base64 avalanche workloads.
#
# The experiment used to run whatever /usr/bin/base64 the machine had, and that
# binary decides the answer.  Measured on one host:
#
#     coreutils 9.11 (Arch pkg,    dynamic)   avalanche 1159/6183  = 18.74%
#     coreutils 9.1  (Debian pkg,  dynamic)   avalanche 1159/6183  = 18.74%
#     coreutils 9.1  (Arch pkg,    dynamic)   avalanche 1061/6645  = 15.97%
#     coreutils 9.11 (from source, static)    avalanche 1146/26721 =  4.29%
#
# These values are illustrative, so the point is not which one is "right": it is
# that the row has to say WHICH binary it measured.  Three are built here so the
# spread is visible rather than accidental.
#
#   base64_debian12  Debian 12's own coreutils package, extracted from the
#                    pinned .deb.  A PREBUILT binary pinned by hash is the only
#                    one of the three that is bit-identical on every machine:
#                    no compiler is involved, so there is no toolchain variance.
#                    This is the reference workload.
#
#   base64_static    Pinned upstream release, built here -O0 -static -no-pie,
#                    the same shape as the nftables and siphash harnesses.
#                    Reproducible GIVEN THE SAME COMPILER: gcc 12.2 and gcc 16.2
#                    produced different binaries (824008 vs 951920 bytes) and
#                    different numbers (4.29% vs 5.40%), because the newer gcc
#                    emits fewer instructions and only the denominator moves.
#                    -static also removes the host libc from the guest, which is
#                    a second source of divergence: Qiling runs with rootfs=/,
#                    so a dynamic guest pulls in the reviewer's libc, and that is
#                    what made this workload crash on Debian 12 but not on Arch.
#
# The third column, the machine's own /usr/bin/base64, needs no build: run-all.sh
# invokes it directly, deliberately unpinned, to show what a reviewer's own
# machine reports.

CURL := curl -sSL

# ---- Debian 12 (bookworm) coreutils, the preinstalled binary ---------------
DEB      := coreutils_9.1-1_amd64.deb
DEB_URL  := http://deb.debian.org/debian/pool/main/c/coreutils/$(DEB)
DEB_SHA  := 61038f857e346e8500adf53a2a0a20859f4d3a3b51570cc876b153a2d51a3091
# sha256 of usr/bin/base64 INSIDE the package, checked after extraction so a
# repackaged .deb with the same name cannot slip a different program through.
BIN_SHA  := ae021af1f99f233eef24c17f9d43843ac36a7c5de1025af794682a89938312e5

# ---- upstream source ------------------------------------------------------
VERSION  := 9.11
TARBALL  := coreutils-$(VERSION).tar.xz
SRC_URL  := https://ftp.gnu.org/gnu/coreutils/$(TARBALL)
SRCDIR   := coreutils-$(VERSION)
SRC_SHA  := 394024eda0a5955217ceda9cd1201e65dc8fa3aa29c2951135a49521d57c3cc3
CFLAGS   := -O0 -static -no-pie -fno-stack-protector

all: base64_debian12 base64_static

$(DEB):
	$(CURL) -O $(DEB_URL)
	echo "$(DEB_SHA)  $(DEB)" | sha256sum -c -

# `ar` comes with binutils, so build-essential already provides it; no dpkg
# needed, which keeps this working on non-Debian hosts.
base64_debian12: $(DEB)
	rm -rf debtmp && mkdir debtmp
	cd debtmp && ar x ../$(DEB) && tar xf data.tar.xz ./usr/bin/base64
	echo "$(BIN_SHA)  debtmp/usr/bin/base64" | sha256sum -c -
	cp debtmp/usr/bin/base64 $@
	rm -rf debtmp

$(TARBALL):
	$(CURL) -O $(SRC_URL)
	echo "$(SRC_SHA)  $(TARBALL)" | sha256sum -c -

$(SRCDIR)/configure: $(TARBALL)
	tar xf $(TARBALL)
	touch $@

# NOTE: build with plain `make`, never `make src/base64`.  Asking coreutils'
# Makefile for that one target skips gnulib's GENERATED headers and the build
# dies on a missing <stdckdint.h>.
base64_static: $(SRCDIR)/configure
	cd $(SRCDIR) && ./configure --quiet --disable-nls --without-selinux \
	    CFLAGS="$(CFLAGS)" LDFLAGS="-static"
	$(MAKE) -C $(SRCDIR)
	cp $(SRCDIR)/src/base64 $@

clean:
	rm -rf $(SRCDIR) debtmp base64_debian12 base64_static

distclean: clean
	rm -f $(TARBALL) $(DEB)

.PHONY: all clean distclean

#!/usr/bin/env bash
## ---------------------------------------------------------------------------
## Downloads and builds TestU01 (and, with --practrand, PractRand) into
## `_deps/`, and compiles the harness `testu01` next to this script.
##
##   ./build.sh                 TestU01 + the harness
##   ./build.sh --practrand     also PractRand's `RNG_test`
##
## Needs curl, a C compiler and make (unzip and g++ for PractRand). Re-running
## it skips the steps whose output already exists; delete `_deps/` to start
## over.
## ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$HERE/_deps"
PREFIX="$DEPS/testu01-install"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# TestU01 1.2.3 (L'Ecuyer & Simard), the authors' repository at a fixed commit
TESTU01_URL="https://github.com/umontreal-simul/TestU01-2009/archive/57e98bf33880daedc930739c81e39b89a8d24dba.tar.gz"
TESTU01_SHA256="8765c359c7cde77731c12d037221a31a29cf4f9e59a9414976cffa2022a91687"
# PractRand pre-0.95 (Doty-Humphrey), the release archive on SourceForge
PRACTRAND_URL="https://downloads.sourceforge.net/project/pracrand/PractRand-pre0.95.zip"
PRACTRAND_SHA256="5738c322ccb983b63fbe693f305afa2a7c73e64a80b303c9139278c453ce27b4"

practrand=0
for arg in "$@"; do
  case "$arg" in
    --practrand) practrand=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DEPS"

# fetch URL SHA256 FILE: downloads URL to FILE, unless it is already there, and
# checks its SHA-256
fetch() {
  local url="$1" sha="$2" file="$3" got
  if [[ ! -f "$file" ]]; then
    echo "downloading $url" >&2
    curl -fsSL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  got="$( (sha256sum "$file" 2>/dev/null || shasum -a 256 "$file") | cut -d' ' -f1)"
  if [[ "$got" != "$sha" ]]; then
    echo "SHA-256 mismatch for $file: expected $sha, got $got" >&2
    rm -f "$file"
    exit 1
  fi
}

# -- TestU01 ------------------------------------------------------------------
if [[ ! -f "$PREFIX/lib/libtestu01.a" ]]; then
  fetch "$TESTU01_URL" "$TESTU01_SHA256" "$DEPS/testu01.tar.gz"
  rm -rf "$DEPS/testu01-src"
  mkdir -p "$DEPS/testu01-src"
  tar -xzf "$DEPS/testu01.tar.gz" -C "$DEPS/testu01-src" --strip-components=1
  src="$DEPS/testu01-src"
  # the repository stores the autotools helper scripts without the executable bit
  chmod +x "$src"/{configure,install-sh,depcomp,missing,mkinstalldirs,config.guess,config.sub}
  echo "building TestU01 in $src" >&2
  # TestU01 (2009) predates GCC 10's -fno-common default and C2x.
  (
    cd "$src"
    CFLAGS="-O2 -fcommon -std=gnu99 -w" ./configure --prefix="$PREFIX" \
      --disable-shared --enable-static > "$DEPS/testu01-configure.log" 2>&1
    make -j"$JOBS" > "$DEPS/testu01-make.log" 2>&1
    make install > "$DEPS/testu01-install.log" 2>&1
  ) || { echo "TestU01 build failed, see $DEPS/testu01-*.log" >&2; exit 1; }
fi

echo "compiling the harness" >&2
cc -O2 -std=gnu99 -Wall -I"$PREFIX/include" -o "$HERE/testu01" "$HERE/testu01.c" \
  "$PREFIX/lib/libtestu01.a" "$PREFIX/lib/libprobdist.a" "$PREFIX/lib/libmylib.a" -lm

# -- PractRand ----------------------------------------------------------------
if [[ "$practrand" == 1 && ! -x "$HERE/RNG_test" ]]; then
  fetch "$PRACTRAND_URL" "$PRACTRAND_SHA256" "$DEPS/practrand.zip"
  rm -rf "$DEPS/practrand-src"
  unzip -q "$DEPS/practrand.zip" -d "$DEPS/practrand-src"
  src="$DEPS/practrand-src"
  echo "building PractRand in $src" >&2
  (
    cd "$src"
    mkdir -p obj
    for f in src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp; do
      g++ -std=c++14 -O3 -w -Iinclude -pthread -c "$f" -o "obj/$(echo "$f" | tr / _).o" &
      while (( $(jobs -r | wc -l) >= JOBS )); do wait -n; done
    done
    wait
    ar rcs libPractRand.a obj/*.o
    g++ -std=c++14 -O3 -w -Iinclude -pthread -o "$HERE/RNG_test" tools/RNG_test.cpp libPractRand.a
  ) > "$DEPS/practrand-build.log" 2>&1 || { echo "PractRand build failed, see $DEPS/practrand-build.log" >&2; exit 1; }
fi

echo "done" >&2

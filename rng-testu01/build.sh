#!/usr/bin/env bash
## ---------------------------------------------------------------------------
## Builds TestU01 (and, with --practrand, PractRand) from the sources in
## `vendor/` into `_deps/`, and compiles the harness `testu01` next to this
## script.
##
##   ./build.sh                 TestU01 + the harness
##   ./build.sh --practrand     also PractRand's `RNG_test`
##
## Needs a C compiler and make (g++ for PractRand). Re-running it skips the
## steps whose output already exists; delete `_deps/` to start over.
## ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor"
DEPS="$HERE/_deps"
PREFIX="$DEPS/testu01-install"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

practrand=0
for arg in "$@"; do
  case "$arg" in
    --practrand) practrand=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DEPS"

# -- TestU01 ------------------------------------------------------------------
if [[ ! -f "$PREFIX/lib/libtestu01.a" ]]; then
  # built in a copy, so that the build leaves `vendor/` untouched
  rm -rf "$DEPS/testu01-src"
  cp -R "$VENDOR/TestU01" "$DEPS/testu01-src"
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
  rm -rf "$DEPS/practrand-src"
  cp -R "$VENDOR/PractRand" "$DEPS/practrand-src"
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

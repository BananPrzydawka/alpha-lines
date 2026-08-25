#!/usr/bin/env bash
#
# Build every kernel in src/, time it, and profile it with Nsight Compute.
# Works unchanged on a rented H100 box or on a laptop with no profiler access.
#
#   ./run.sh                 # auto-detect the local GPU's arch
#   ARCH=sm_90a ./run.sh     # force an arch (needed on H100: auto gives sm_90)
#   SET=detailed ./run.sh    # lighter ncu metric set; see `ncu --list-sets`
#   NCU="sudo ncu" ./run.sh  # run the profiler as root

# ---------------------------------------------------------------------------
# Shell safety options. Without these, bash silently plows through failures.
#   -e  abort the whole script the moment any command exits non-zero
#   -u  abort if we ever read a variable that was never set (catches typos)
#   -o pipefail  make `a | b` fail if EITHER side fails, not just the last one.
#               Without it, `./add | tee f` reports success even if ./add died.
# ---------------------------------------------------------------------------
set -euo pipefail

# $0 is the path this script was invoked as; dirname strips the filename.
# So this cd's into the script's own directory, letting you run it from
# anywhere: `~/ $ ~/projects/alpha-lines/cuda/run.sh` still works.
cd "$(dirname "$0")"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ${VAR:-default} means "the value of VAR, or `default` if VAR is unset/empty".
# So `SET=detailed ./run.sh` overrides this; a bare `./run.sh` gets "full".
SET=${SET:-full}

# Binaries that should be timed but NOT profiled. Space-separated names.
# latency.cu measures with clock64() inside a single warp -- ncu counters
# are meaningless for it, and profiling it just wastes time.
NOPROFILE=${NOPROFILE:-""}

# MK holds the extra arguments we pass to make. It's an ARRAY (the parens),
# not a string, so each element stays one argv entry even if it had spaces.
#   - if ARCH is unset  -> MK stays empty  -> make auto-detects the arch
#   - if ARCH is set    -> MK is ("ARCH=sm_90a") -> make uses that
# `${ARCH:-}` is written with the :- so that `set -u` doesn't abort when
# ARCH was never defined. `[[ -n X ]]` is "X is a non-empty string".
MK=()
[[ -n ${ARCH:-} ]] && MK=(ARCH="$ARCH")


# ---------------------------------------------------------------------------
# Work out where things live
# ---------------------------------------------------------------------------

# $(cmd) is command substitution: run cmd, paste its stdout here.
# We ask the Makefile where its build directory is rather than recomputing
# the arch-detection logic in bash. One source of truth.
#   make -s   = silent, don't echo recipes, so only the `echo` reaches us
BLD=$(make -s printbld "${MK[@]}")

# Name the output directory after the GPU, e.g. NVIDIA_GeForce_GTX_1660_Ti.
#   2>/dev/null  discard nvidia-smi's error text if there's no GPU
#   head -1      keep only the first GPU in a multi-GPU box
#   tr ' ' '_'   spaces -> underscores, so it's a usable directory name
#   || true      stop `set -e` from killing us when nvidia-smi is missing
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
      | head -1 | tr ' ' '_' || true)
OUT=out/${TAG:-unknown}
mkdir -p "$OUT"


# ---------------------------------------------------------------------------
# Build. `bins` makes the executables, `inspect` makes .ptx/.sass/ptxas.txt.
# Both land in $BLD. Everything below consumes what this produces.
# ---------------------------------------------------------------------------
make bins inspect "${MK[@]}"


# ---------------------------------------------------------------------------
# Collect the list of executables to run.
#
# $BLD contains a mix: `add` (executable) alongside `add.ptx`, `add.sass`,
# `add.cubin` (not executable). We want only the binaries.
#   -x $k        true if $k is executable
#   ! $k =~ \.   true if the path contains NO dot  (=~ is regex match)
# Both must hold. This is why the Makefile strips the dot out of the arch
# string -- "sm_7.5" in the path would break this filter.
# ---------------------------------------------------------------------------
BINS=()
for k in "$BLD"/*; do
    [[ -x $k && ! $k =~ \. ]] || continue   # `continue` = skip to next loop item
    BINS+=("$k")                            # append to the array
done

# ${#BINS[@]} is "number of elements in BINS".
if [[ ${#BINS[@]} -eq 0 ]]; then
    echo "no executables found in $BLD" >&2   # >&2 writes to stderr
    exit 1
fi


# ---------------------------------------------------------------------------
# Pass 1: wall-clock timing.
# Real boost clocks, warm caches, steady state -- what the kernel actually
# costs in a running program. This is the number to trust for "how fast".
# ---------------------------------------------------------------------------
for k in "${BINS[@]}"; do
    n=$(basename "$k")            # build/sm_75/add  ->  add
    echo "== $n"
    # "$k" on its own line EXECUTES the binary at that path.
    # tee writes stdout to a file AND passes it through to your terminal,
    # which is why you see the numbers live and they're also saved.
    "$k" | tee "$OUT/$n.txt"
done


# ---------------------------------------------------------------------------
# Pass 2: hardware counters via Nsight Compute.
# Locked base clocks and flushed L2 -- reproducible ratios, NOT wall time.
# Use pass 1 for duration, pass 2 for "why".
#
# `command -v ncu` prints ncu's path if it exists, fails if it doesn't.
# ---------------------------------------------------------------------------
if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not installed -- timings only"
    exit 0                        # not an error: the timings above are valid
fi

fail=0
for k in "${BINS[@]}"; do
    n=$(basename "$k")

    # Skip anything listed in NOPROFILE. The pattern below is a substring
    # test with spaces around both sides, so "lat" doesn't match "latency".
    if [[ " $NOPROFILE " == *" $n "* ]]; then
        echo "== $n (profiling skipped)"
        continue
    fi

    echo "== $n (ncu, set=$SET)"

    # set +e / set -e temporarily disables abort-on-error, so that a failing
    # ncu lets us print a useful hint instead of killing the script silently.
    #   --profile-from-start off  only profile between cudaProfilerStart/Stop
    #   --import-source yes       embed the .cu source into the report so the
    #                             Source page can map SASS back to your lines
    #   -f                        overwrite an existing report file
    #   -o                        report path (ncu appends .ncu-rep)
    #   trailing 200              argv[1] for the binary itself (iteration count)
    set +e
    ${NCU:-ncu} --profile-from-start off \
                --set "$SET" \
                --import-source yes \
                -f -o "$OUT/$n" \
                "$k" 200
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        echo "  ncu failed on $n (rc=$rc)"
        fail=1
        continue
    fi

    # -i reads an existing report instead of profiling something new.
    # These dumps let you read counters over SSH without the ncu-ui GUI.
    ncu -i "$OUT/$n.ncu-rep" --page details      > "$OUT/${n}_details.txt"
    ncu -i "$OUT/$n.ncu-rep" --page raw --csv    > "$OUT/${n}_raw.csv"
    # The Source page needs -lineinfo at compile time; `|| true` because it
    # legitimately has nothing to show for some kernels.
    ncu -i "$OUT/$n.ncu-rep" --page source --print-source sass \
                                                 > "$OUT/${n}_sass.txt" 2>/dev/null || true
done

if [[ $fail -ne 0 ]]; then
    echo
    echo "At least one profile failed."
    echo "  ERR_NVGPUCTRPERM means the driver is blocking counter access. Fix:"
    echo "    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\"
    echo "      | sudo tee /etc/modprobe.d/nvidia-profiler.conf"
    echo "    sudo mkinitcpio -P        # Arch    (Debian/Ubuntu: update-initramfs -u)"
    echo "    sudo reboot"
    echo "  Or run this script as: NCU='sudo ncu' ./run.sh"
fi

echo "--> $OUT"
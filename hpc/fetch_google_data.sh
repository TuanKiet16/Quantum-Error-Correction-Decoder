#!/usr/bin/env bash
# Fetch the Google 105-qubit surface-code dataset (d3/d5/d7) from Zenodo.
# Part 6 (optional) input for qec_decoder.google_data. ~5.7 GB.
# Run on HPC scratch, not locally. Resumable; verifies md5.
set -euo pipefail

DEST_DIR="${1:-data}"
FILE="google_105Q_surface_code_d3_d5_d7.zip"
URL="https://zenodo.org/api/records/13273331/files/${FILE}/content"
MD5="21fa6ad35b395d838ebcdbc92e364a12"
SIZE=5716907033

mkdir -p "$DEST_DIR"
OUT="${DEST_DIR}/${FILE}"

echo "Downloading ${FILE} (~5.7 GB) to ${OUT}"
# -C - resumes a partial download; retries on transient network drops.
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$OUT" "$URL"

echo "Verifying md5..."
if command -v md5sum >/dev/null 2>&1; then
    GOT="$(md5sum "$OUT" | awk '{print $1}')"
else
    GOT="$(md5 -q "$OUT")"  # macOS
fi

if [ "$GOT" != "$MD5" ]; then
    echo "CHECKSUM MISMATCH: got $GOT expected $MD5" >&2
    echo "Delete $OUT and re-run." >&2
    exit 1
fi

echo "OK: ${OUT}"
echo "md5 $GOT  size $(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT") (expected $SIZE)"
echo "qec_decoder.google_data.GOOGLE_ZIP expects this at data/${FILE}"

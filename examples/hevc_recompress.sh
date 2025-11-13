#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/.." && pwd)"
LIST="${PD}/test_video_names.txt"

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <crf> <scale> <in_dir> <jobs> <num>" >&2
  exit 2
fi

CRF="$1"
SCALE="$2"
IN_DIR="${3%/}"
JOBS="$4"
NUM="$5"

TMPDIR="$(mktemp -d)"
OUT_ZIP="$PD/comma2k19_submission.zip"

export CRF IN_DIR TMPDIR SCALE

head -n "$NUM" "$LIST" | xargs -n1 -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  OUT="${TMPDIR}/${rel}"
  mkdir -p "$(dirname "$OUT")"

  echo "→ ${IN}  CRF=${CRF}  scale=${SCALE}  →  ${OUT}"

  if [[ "$SCALE" =~ ^1(\.0+)?$ ]]; then
    SCALE_VF=""
  else
    SCALE_VF="-vf scale=trunc(iw*${SCALE}/2)*2:trunc(ih*${SCALE}/2)*2:flags=lanczos"
  fi

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    ${SCALE_VF} \
    -c:v libx265 -preset fast -crf "$CRF" \
    -g 1 -bf 0 -x265-params "keyint=1:min-keyint=1:scenecut=0:frame-threads=1:log-level=warning" \
    -r 20 -f hevc "$OUT"
' _ {}

rm -f "$OUT_ZIP"
(
  cd "$TMPDIR"
  zip -r "$OUT_ZIP" .
)
rm -rf "$TMPDIR"
echo "All done. Saved $OUT_ZIP"

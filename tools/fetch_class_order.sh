#!/usr/bin/env bash
#
# Fetch a NOAA CLASS order into a destination directory.
#
# CLASS throttles per connection rather than per client, so a sequential loop
# runs at roughly 200 KB/s while several concurrent streams each sustain the
# same rate. A few parallel streams therefore multiply throughput. Four is the
# default: enough to matter on a multi-gigabyte order, few enough to stay
# polite to a shared public archive.
#
# Large granules on a slow link can exceed any fixed timeout, so this does NOT
# impose a total time limit. It aborts a transfer only when it genuinely
# stalls (under 10 KB/s for 60 seconds) and resumes from wherever it stopped
# rather than starting over. Files already complete are skipped by comparing
# local size with the server's Content-Length, so the script is safe to re-run
# and safe to interrupt.
#
# Usage:
#   bash tools/fetch_class_order.sh ORDER_URL DEST_DIR [PARALLEL]
#   bash tools/fetch_class_order.sh --list URLFILE DEST_DIR [PARALLEL]
#
# ORDER_URL is the public order directory, for example
# https://order.class.noaa.gov/public/8564266452/ (the 001/ subdirectory is
# found automatically).
#
set -euo pipefail

if [ "${1:-}" = "--list" ]; then
  URLFILE="${2:?need a URL list file}"
  DEST="${3:?need a destination directory}"
  PAR="${4:-4}"
else
  ORDER="${1:?need an order URL}"
  DEST="${2:?need a destination directory}"
  PAR="${3:-4}"
  URLFILE="$(mktemp)"
  base="${ORDER%/}"
  for sub in "$base/001/" "$base/"; do
    curl -s --max-time 120 "$sub" \
      | grep -oE '/downloads/public/[^"]+\.(tar|nc|gz)' \
      | sed 's#^#https://order.class.noaa.gov#' | sort -u >> "$URLFILE" || true
    [ -s "$URLFILE" ] && break
  done
fi

total=$(wc -l < "$URLFILE" | tr -d ' ')
[ "$total" -gt 0 ] || { echo "no files found in the order listing" >&2; exit 1; }
mkdir -p "$DEST"
echo "$total file(s) listed, $PAR parallel streams, into $DEST"

fetch_one() {
  url="$1"; dest="$2"
  f="$dest/$(basename "$url")"
  remote=$(curl -sI --max-time 60 "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{gsub("\r","");print $2}' | tail -1)
  if [ -n "${remote:-}" ] && [ -f "$f" ]; then
    local_size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$local_size" = "$remote" ]; then
      echo "have $(basename "$f")"
      return 0
    fi
  fi
  # Resume from wherever a previous attempt stopped; abort only on a real
  # stall, never on a merely slow link.
  if curl -sS --fail --location -C - \
        --retry 5 --retry-delay 10 --retry-all-errors \
        --speed-time 60 --speed-limit 10240 \
        -o "$f" "$url"; then
    got=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ -n "${remote:-}" ] && [ "$got" != "$remote" ]; then
      echo "SHORT $(basename "$f") ($got of $remote bytes, re-run to resume)"
      return 1
    fi
    echo "ok   $(basename "$f")"
  else
    echo "FAIL $(basename "$f") (re-run to resume)"
    return 1
  fi
}
export -f fetch_one

# xargs returns non-zero if any child failed; report rather than abort, since a
# re-run resumes cleanly.
set +e
xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {} "$DEST" < "$URLFILE"
set -e

done_n=0
while read -r u; do
  f="$DEST/$(basename "$u")"
  [ -s "$f" ] && done_n=$((done_n + 1))
done < "$URLFILE"
echo "complete: $done_n of $total present in $DEST"

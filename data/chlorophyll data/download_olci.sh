#!/bin/bash
set -uo pipefail

TOKEN_FILE="$HOME/.edl_token"
OUT_DIR="/Users/tushy/Documents/Research/Ghana Project/data/chlorophyll data"
LOG="$OUT_DIR/download_olci.log"
PARALLEL=6

start_date="2025-10-05"
end_date="2026-02-11"

download_day() {
    local d="$1"
    local ds
    ds=$(echo "$d" | tr -d '-')
    local token
    token=$(cat "$TOKEN_FILE")

    for sensor in S3A_OLCI_ERRNT S3B_OLCI_ERRNT; do
        local fname="${sensor}.${ds}.L3m.DAY.CHL.chlor_a.4km.nc"
        local out="$OUT_DIR/$fname"
        if [[ -f "$out" ]]; then
            continue
        fi
        local code
        code=$(curl -s -o "$out" -w "%{http_code}" --header "Authorization: Bearer $token" "https://oceandata.sci.gsfc.nasa.gov/getfile/$fname")
        if [[ "$code" == "200" ]]; then
            echo "OK $fname" >> "$LOG"
        else
            echo "FAILED($code) $fname" >> "$LOG"
            rm -f "$out"
        fi
    done
}
export -f download_day
export TOKEN_FILE OUT_DIR LOG

: > "$LOG"

d="$start_date"
while [[ "$d" != $(date -j -v+1d -f "%Y-%m-%d" "$end_date" "+%Y-%m-%d") ]]; do
    echo "$d"
    d=$(date -j -v+1d -f "%Y-%m-%d" "$d" "+%Y-%m-%d")
done | xargs -P "$PARALLEL" -I{} bash -c 'download_day "$@"' _ {}

echo "DONE" >> "$LOG"

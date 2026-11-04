#!/bin/sh
set -eu

if [ -d /app/deploy/seed ]; then
    mkdir -p /app/data/chunks
    for strategy in section fixed; do
        cp "/app/deploy/seed/chunks_${strategy}.jsonl" /app/data/chunks/
    done
    biq index --config hybrid --reuse-chunks
    biq index --config dense_fixed --reuse-chunks
else
    biq fetch
    biq parse --no-ocr
    biq index --config hybrid
    biq index --config dense_fixed
fi

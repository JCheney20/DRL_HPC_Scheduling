#!/usr/bin/env bash
set -euo pipefail
base="/scratch/${USER}/DRL_HPC_Scheduling"
echo "Redirecting outputs to $base ..."
mkdir -p "$base/trained_model" "$base/result" "$base/logs"
for d in trained_model result logs; do
    if [ -L "$d" ]; then
        echo "  $d already -> $(readlink "$d")"
    else
        if [ -e "$d" ]; then
            echo "  migrating existing $d/ -> $base/$d/"
            rsync -a "$d"/ "$base/$d"/
            rm -rf "$d"
        fi
        ln -s "$base/$d" "$d"
        echo "  linked $d -> $base/$d"
    fi
done
echo "✓ Outputs now on $base. Archive final models at the end with: just archive_results"

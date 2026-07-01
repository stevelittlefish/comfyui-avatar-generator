#!/bin/bash
set -euo pipefail
# A script to prepare the vessel and set sail with the slop!

VENV_DIR="venv"
REQ_FILE="requirements.txt"
REQ_STAMP="$VENV_DIR/.requirements.stamp"
REINSTALL=0
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --reinstall)
            REINSTALL=1
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found. Forging a new one... Arrr!"
    python3 -m venv "$VENV_DIR"
else
    echo "Venv found. Ready to sail!"
fi

if [ "$REINSTALL" -eq 1 ] || [ ! -f "$REQ_STAMP" ] || [ "$REQ_FILE" -nt "$REQ_STAMP" ]; then
    echo "Installing requirements... mind the barnacles."
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
    touch "$REQ_STAMP"
else
    echo "Requirements already shipshape."
fi

# Launch the slop generator
"$VENV_DIR/bin/python" avatar_gen.py "${ARGS[@]}"

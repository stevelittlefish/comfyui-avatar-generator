#!/bin/bash
# A script to prepare the vessel and set sail with the slop!

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found. Forging a new one... Arrr!"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -r requirements.txt
else
    echo "Venv found. Ready to sail!"
fi

# Launch the slop generator
"$VENV_DIR/bin/python" avatar_gen.py "$@"

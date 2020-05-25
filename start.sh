#!/usr/bin/env bash

# shellcheck disable=SC1091

[[ -d "venv" ]] || python3.8 -m venv ./venv
source venv/bin/activate
source .env
pip install -r requirements.txt
gunicorn charmboard:app -b :5500 --workers=8 --reload
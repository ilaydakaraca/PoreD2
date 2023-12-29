#!/usr/bin/env bash
# exit on error
set -o errexit

python3.9 -m pip install -r requirements.txt

python manage.py collectstatic --no-input

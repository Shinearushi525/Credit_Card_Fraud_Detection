#!/usr/bin/env bash
set -e
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
streamlit run app.py --server.port 3000 --server.address 0.0.0.0 --server.headless true

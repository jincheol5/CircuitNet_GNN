#!/bin/bash
set -e

python -m app.dataset.check_dataset --app_num 1 --data_type nodes
python -m app.dataset.check_dataset --app_num 1 --data_type net_edges
python -m app.dataset.check_dataset --app_num 1 --data_type pin_positions

python -m app.dataset.check_dataset --app_num 2 
python -m app.dataset.check_dataset --app_num 2 
python -m app.dataset.check_dataset --app_num 2 
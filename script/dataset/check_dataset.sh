#!/bin/bash
set -e

python -m app.dataset.check_dataset --app_num 1 --data_type nodes
python -m app.dataset.check_dataset --app_num 1 --data_type net_edges
python -m app.dataset.check_dataset --app_num 1 --data_type pin_positions

python -m app.dataset.check_dataset --app_num 2 --file_name 293-RISCY-a-1-c5-u0.75-m2-p6-f0

python -m app.dataset.check_dataset --app_num 3

python -m app.dataset.check_dataset --app_num 4

python -m app.dataset.check_dataset --app_num 5
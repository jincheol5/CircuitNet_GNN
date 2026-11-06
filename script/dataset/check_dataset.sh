#!/bin/bash
set -e

python app.dataset.check_dataset --app_num 1 --data_type nodes
python app.dataset.check_dataset --app_num 1 --data_type net_edges
python app.dataset.check_dataset --app_num 1 --data_type pin_positions

python app.dataset.check_dataset --app_num 2 
python app.dataset.check_dataset --app_num 2 
python app.dataset.check_dataset --app_num 2 
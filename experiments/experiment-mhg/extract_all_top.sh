#!/bin/sh

python extract_top.py results_test/MHG_results_base.json results_test/MHG_top_base.json 100
python extract_top.py results_test/MHG_results_I1.json results_test/MHG_top_I1.json 100
python extract_top.py results_test/MHG_results_I2.json results_test/MHG_top_I2.json 100
python extract_top.py results_test/MHG_results_I3.json results_test/MHG_top_I3.json 100
python extract_top.py results_test/MHG_results_I4.json results_test/MHG_top_I4.json 100
python extract_top.py results_test/MHG_results_I5.json results_test/MHG_top_I5.json 100
python extract_top.py results_test/MHG_results_I6.json results_test/MHG_top_I6.json 100

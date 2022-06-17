#!/bin/sh

python rate_test.py models/model_MHG_base.pth results_test/MHG_results_base.json
python rate_test.py models/models_MHG_I1.pth results_test/MHG_results_I1.json
python rate_test.py models/models_MHG_I2.pth results_test/MHG_results_I2.json
python rate_test.py models/models_MHG_I3.pth results_test/MHG_results_I3.json
python rate_test.py models/models_MHG_I4.pth results_test/MHG_results_I4.json
python rate_test.py models/models_MHG_I5.pth results_test/MHG_results_I5.json
python rate_test.py models/models_MHG_I6.pth results_test/MHG_results_I6.json

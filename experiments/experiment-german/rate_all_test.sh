#!/bin/sh

python rate_test.py models/model_GER_base.pth results_test/results_base.json
python rate_test.py models/models_GER_I1 results_test/results_I1.json
python rate_test.py models/models_GER_I2.pth results_test/results_I2.json
python rate_test.py models/models_GER_I3.pth results_test/results_I3.json
python rate_test.py models/models_GER_I4.pth results_test/results_I4.json
python rate_test.py models/models_GER_I5.pth results_test/results_I5.json
python rate_test.py models/models_GER_I6.pth results_test/results_I6.json



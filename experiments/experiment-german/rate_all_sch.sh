#!/bin/sh

python rate_sch.py models/model_GER_base.pth results_sch/results_sch_base.json
python rate_sch.py models/models_GER_I1 results_sch/results_sch_I1.json
python rate_sch.py models/models_GER_I2.pth results_sch/results_sch_I2.json
python rate_sch.py models/models_GER_I3.pth results_sch/results_sch_I3.json
python rate_sch.py models/models_GER_I4.pth results_sch/results_sch_I4.json
python rate_sch.py models/models_GER_I5.pth results_sch/results_sch_I5.json
python rate_sch.py models/models_GER_I6.pth results_sch/results_sch_I6.json



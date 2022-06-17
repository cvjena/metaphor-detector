#!/bin/sh

echo "top_base_annotated"
python get_t100_precision.py annotations_test/MHG_top_base_annotated.json

echo "top_I1_annotated"
python get_t100_precision.py annotations_test/MHG_top_I1_annotated.json

echo "top_I2_annotated"
python get_t100_precision.py annotations_test/MHG_top_I2_annotated.json

echo "top_I3_annotated"
python get_t100_precision.py annotations_test/MHG_top_I3_annotated.json

echo "top_I4_annotated"
python get_t100_precision.py annotations_test/MHG_top_I4_annotated.json

echo "top_I5_annotated"
python get_t100_precision.py annotations_test/MHG_top_I5_annotated.json

echo "top_I6_annotated"
python get_t100_precision.py annotations_test/MHG_top_I6_annotated.json

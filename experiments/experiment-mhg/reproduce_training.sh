#!/bin/sh
echo "get vector pairs"
#python get_vector_pairs.py ADJ NA
echo "diff vector pairs"
for fn in ./vector_pairs/* ; do
      
    fnout=./vector_pairs.paper/`echo $fn | cut -d "/" -f3`
    diff $fn $fnout
done
echo "done"

echo "train network"
# this takes 23 epochs
#python ./train_network_from_texts.py
echo "diff models"
#torchdiff models/model_MHG_base.pth models.paper/model_MHG_base.pth
echo "done"

echo "rate base"
#python rate.py models/model_MHG_base.pth results/results_base.json
echo "diff"
#diff results/results_base.json results.paper/results_base.json
echo "done"

echo "extract chosen base"
#python extract_chosen.py results/results_base.json results/chosen_base.json
echo "diff"
#diff results/chosen_base.json results.paper/chosen_base.json
echo "done"

echo "create base dataset"
#python annotations_to_datasets.py datasets/MHG_from_base.pickle annotations/MHG_chosen_base_annoateted.json
echo "diff"
#diff datasets/MHG_from_base.pickle datasets.paper/MHG_from_base.pickle
echo "done"


echo "finetune"
# 14 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I1.pth datasets/MHG_from_base.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"
echo "rate"
#python rate.py models/models_MHG_I1.pth results/MHG_results_F1.json
echo "diff"
#diff results/MHG_results_f1.json results.paper/MHG_results_f1.json
echo "done"
echo "extract chosen"
#python extract_chosen_ann.py results/MHG_results_F1.json results/MHG_chosen_F1.json annotations/MHG_chosen_base_annoateted.json 
echo "diff"
#diff results/MHG_chosen_F1.json results.paper/MHG_chosen_F1.json
echo "done"
echo "apply annotations"
#python annotations_to_datasets.py datasets/MHG_from_F1.pickle annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json
echo "diff"
#diff datasets/MHG_from_F1.pickle datasets.paper/MHG_from_F1.pickle
echo "done"


echo "finetune"
# 3 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I2.pth datasets/MHG_from_F1.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"
echo "rate"
#python rate.py models/models_MHG_I2.pth results/MHG_results_F2.json
echo "diff"
#diff results/MHG_results_F2.json results.paper/MHG_results_F2.json
echo "done"
echo "extract chosen"
#python extract_chosen_ann.py results/MHG_results_F2.json results/MHG_chosen_F2.json annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json
echo "diff"
#diff results/MHG_chosen_F2.json results.paper/MHG_chosen_F2.json
echo "done"
echo "apply annotations"
#python annotations_to_datasets.py datasets/MHG_from_F2.pickle annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json
echo "diff"
#diff datasets/MHG_from_F2.pickle datasets.paper/MHG_from_F2.pickle
echo "done"

echo "finetune"
# 4 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I3.pth datasets/MHG_from_F2.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"
echo "rate"
#python rate.py models/models_MHG_I3.pth results/MHG_results_F3.json
echo "diff"
#diff results/MHG_results_F3.json results.paper/MHG_results_F3.json
echo "done"
echo "extract chosen"
#python extract_chosen_ann.py results/MHG_results_I3.json results/MHG_chosen_I3.json annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json 
echo "diff"
#diff results/MHG_chosen_I3.json results.paper/MHG_chosen_I3.json
echo "done"
echo "apply annotations"
#python annotations_to_datasets.py datasets/MHG_from_F3.pickle annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json annotations/MHG_chosen_I3_annotated.json
echo "diff"
#diff datasets/MHG_from_F3.pickle datasets.paper/MHG_from_F3.pickle
echo "done"


echo "finetune"
# 5 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I4.pth datasets/MHG_from_F3.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"
echo "rate"
#python rate.py models/models_MHG_I4.pth results/MHG_results_I4.json
echo "diff"
#diff results/MHG_results_I4.json results.paper/MHG_results_I4.json
echo "done"
echo "extract chosen"
#python extract_chosen_ann.py results/MHG_results_I4.json results/MHG_chosen_I4.json annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json annotations/MHG_chosen_I3_annotated.json
echo "diff"
#diff results/MHG_chosen_I4.json results.paper/MHG_chosen_I4.json
echo "done"
echo "apply annotations"
#python annotations_to_datasets.py datasets/MHG_from_F4.pickle annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json annotations/MHG_chosen_I3_annotated.json annotations/MHG_chosen_I4_annotated.json
echo "diff"
#diff datasets/MHG_from_F4.pickle datasets.paper/MHG_from_F4.pickle
echo "done"


echo "finetune"
# 6 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I5.pth datasets/MHG_from_F4.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"
echo "rate"
#python rate.py models/models_MHG_I5.pth results/MHG_results_I5.json
echo "diff"
#diff results/MHG_results_I5.json results.paper/MHG_results_I5.json
echo "done"
echo "extract chosen"
#python extract_chosen_ann.py results/MHG_results_I5.json results/MHG_chosen_I5.json annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json annotations/MHG_chosen_I3_annotated.json annotations/MHG_chosen_I4_annotated.json
echo "diff"
#diff results/MHG_chosen_I5.json results.paper/MHG_chosen_I5.json
echo "done"
echo "apply annotations"
#python annotations_to_datasets.py datasets/MHG_from_I5.pickle annotations/MHG_chosen_base_annoateted.json annotations/MHG_chosen_F1_annotated.json annotations/MHG_chosen_F2_annotated.json annotations/MHG_chosen_I3_annotated.json annotations/MHG_chosen_I4_annotated.json annotations/MHG_chosen_I5_annotated.json
echo "diff"
#diff datasets/MHG_from_I5.pickle datasets.paper/MHG_from_I5.pickle
echo "done"

echo "finetune"
# 6 epochs
#python finetune_network_on_datasets.py models/model_MHG_base.pth models/models_MHG_I6.pth datasets/MHG_from_I5.pickle
echo "diff models"
#torchdiff models/models_MHG_I1.pth models.paper/models_MHG_I1.pth
echo "done"

echo "rate all test"
#sh rate_all_test.sh
echo "extract all top"
sh extract_all_top.sh

echo "diff test resutlts"
for fn in ./results_test/* ; do
      
    fnout=./results_test.paper/`echo $fn | cut -d "/" -f3`
    diff $fn $fnout
done
echo "done"

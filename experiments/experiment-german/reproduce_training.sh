#!/bin/sh
echo "train model"
#python train_network_from_texts.py
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/model_GER_base.pth models.paper/model_GER_base.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I1 datasets/from_base.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I1 models.paper/models_GER_I1


echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I2.pth datasets/from_F1.pickle 
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I2.pth models.paper/models_GER_I2.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I3.pth datasets/from_F2.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I3.pth models.paper/models_GER_I3.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I3.pth datasets/from_F2.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I3.pth models.paper/models_GER_I3.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I4.pth datasets/from_F3.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I4.pth models.paper/models_GER_I4.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I5.pth datasets/from_F4.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I5.pth models.paper/models_GER_I5.pth

echo "finetune model"
#python finetune_network_on_datasets.py models/model_GER_base.pth models/models_GER_I6.pth datasets/from_F5.pickle
echo "torchdiff"
#./torchdiff /scripts/torchdiff models/models_GER_I6.pth models.paper/models_GER_I6.pth


echo "results_base.json"
#python rate.py models/model_GER_base.pth results/results_base.json
echo "diff test"
#diff results/results_base.json results.paper/results_base.json
echo "diff test done\n"
echo "results_F1.json"
#python rate.py models/models_GER_I1 results/results_F1.json
echo "diff test"
#diff results/results_F1.json results.paper/results_F1.json
echo "diff test done\n"
echo "results_F2.json"
#python rate.py models/models_GER_I2.pth results/results_F2.json
echo "diff test"
#diff results/results_F2.json results.paper/results_F2.json
echo "diff test done\n"
echo "results_F3.json"
#python rate.py models/models_GER_I3.pth results/results_F3.json
echo "diff test"
#diff results/results_F3.json results.paper/results_F3.json
echo "diff test done\n"
echo "results_F4.json"
#python rate.py models/models_GER_I4.pth results/results_F4.json
echo "diff test"
#diff results/results_F4.json results.paper/results_F4.json
echo "diff test done\n"
echo "results_F5.json"
#python rate.py models/models_GER_I5.pth results/results_F5.json
echo "diff test"
#diff results/results_F5.json results.paper/results_F5.json
echo "diff test done\n"
echo "done"


echo "diff the chosen values"
echo "chosen_base.json"
#python extract_chosen.py results/results_base.json results/chosen_base.json
#diff results/chosen_base.json results.paper/chosen_base.json

echo "chosen_F1.json"
#python extract_chosen_ann.py results/results_F1.json results/chosen_F1.json annotations/chosen_base_annotated.json
#diff results/chosen_F1.json results.paper/chosen_F1.json

echo "chosen_F2.json"
#python extract_chosen_ann.py results/results_F2.json results/chosen_F2.json annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json
#diff results/chosen_F2.json results.paper/chosen_F2.json

echo "chosen_F3.json"
#python extract_chosen_ann.py results/results_F3.json results/chosen_F3.json annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json
#diff results/chosen_F3.json results.paper/chosen_F3.json

echo "chosen_F4.json"
#python extract_chosen_ann.py results/results_F4.json results/chosen_F4.json annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json annotations/chosen_F3_annotated.json
#diff results/chosen_F4.json results.paper/chosen_F4.json

echo "chosen_F5.json"
#python extract_chosen_ann.py results/results_F5.json results/chosen_F5.json annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json annotations/chosen_F3_annotated.json annotations/chosen_F4_annotated.json
#diff results/chosen_F5.json results.paper/chosen_F5.json
echo "done"


echo "annotations to datasets"
echo "creating from_base.pickle"
#python annotations_to_datasets.py datasets/from_base.pickle annotations/chosen_base_annotated.json
echo "diffing from_base.pickle"
#diff datasets/from_base.pickle datasets.paper/from_base.pickle
echo "creating from_F1.pickle"
#python annotations_to_datasets.py annotations/from_F1.pickle annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json
echo "diffing from_F1.pickle"
#diff datasets/from_F1.pickle datasets.paper/from_F1.pickle
echo "creating from_F2.pickle"
#python annotations_to_datasets.py datasets/from_F2.pickle annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json 
echo "diffing from_F2.pickle"
#diff datasets/from_F2.pickle datasets.paper/from_F2.pickle
echo "creating from_F3.pickle"
#python annotations_to_datasets.py datasets/from_F3.pickle annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json annotations/chosen_F3_annotated.json
echo "diffing from_F3.pickle"
#diff datasets/from_F3.pickle datasets.paper/from_F3.pickle
echo "creating from_F4.pickle"
#python annotations_to_datasets.py datasets/from_F4.pickle annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json annotations/chosen_F3_annotated.json annotations/chosen_F4_annotated.json
echo "diffing from_F4.pickle"
#diff datasets/from_F4.pickle datasets.paper/from_F4.pickle
echo "creating from_F5.pickle"
#python annotations_to_datasets.py datasets/from_F5.pickle annotations/chosen_base_annotated.json annotations/chosen_F1_annotated.json annotations/chosen_F2_annotated.json annotations/chosen_F3_annotated.json annotations/chosen_F4_annotated.json annotations/chosen_F5_annotated.json 
echo "diffing from_F5.pickle"
#diff datasets/from_F5.pickle datasets.paper/from_F5.pickle
echo "done"

echo "rating test"
#sh rate_all_test.sh
echo "diffing test results"
#diff results_test/results_base.json results_test.paper/results_base.json
#diff results_test/results_I1.json results_test.paper/results_I1.json
#diff results_test/results_I2.json results_test.paper/results_I2.json
#diff results_test/results_I3.json results_test.paper/results_I3.json
#diff results_test/results_I4.json results_test.paper/results_I4.json
#diff results_test/results_I5.json results_test.paper/results_I5.json
#diff results_test/results_I6.json results_test.paper/results_I6.json
echo "done"

echo "choose top 100 test examples"
#python extract_top.py results_test/results_base.json results_test/top_base.json 100
#python extract_top.py results_test/results_I1.json results_test/top_I1.json 100
#python extract_top.py results_test/results_I2.json results_test/top_I2.json 100
#python extract_top.py results_test/results_I3.json results_test/top_I3.json 100
#python extract_top.py results_test/results_I4.json results_test/top_I4.json 100
#python extract_top.py results_test/results_I5.json results_test/top_I5.json 100
#python extract_top.py results_test/results_I6.json results_test/top_I6.json 100

echo "diff top 100"
#diff results_test/top_base.json results_test.paper/top_base.json
#diff results_test/top_I1.json results_test.paper/top_I1.json
#diff results_test/top_I2.json results_test.paper/top_I2.json
#diff results_test/top_I3.json results_test.paper/top_I3.json
#diff results_test/top_I4.json results_test.paper/top_I4.json
#diff results_test/top_I5.json results_test.paper/top_I5.json
#diff results_test/top_I6.json results_test.paper/top_I6.json
echo "done"

echo "### schiller ###"
echo "get pos"
#sh get_pos_sch.sh
echo "get vectors"
#sh get_vectors_sch.sh
echo "get sentences"
#sh get_sentence_boundaries_sch.sh
echo "get vector_pairs"
python get_vector_pairs_sch.py ADJ NOUN

echo "rating schiller"
sh rate_all_sch.sh
echo "extracting schiller top"
python extract_top.py results_sch/results_sch_base.json results_sch/top_sch_base.json 100
python extract_top.py results_sch/results_sch_I1.json results_sch/top_sch_I1.json 100
python extract_top.py results_sch/results_sch_I2.json results_sch/top_sch_I2.json 100
python extract_top.py results_sch/results_sch_I3.json results_sch/top_sch_I3.json 100
python extract_top.py results_sch/results_sch_I4.json results_sch/top_sch_I4.json 100
python extract_top.py results_sch/results_sch_I5.json results_sch/top_sch_I5.json 100
python extract_top.py results_sch/results_sch_I6.json results_sch/top_sch_I6.json 100

echo "diff schiller results"
for fn in ./results_sch/* ; do
      
    fnout=./results_sch.paper/`echo $fn | cut -d "/" -f3`
    diff $fn $fnout
done
echo "done"

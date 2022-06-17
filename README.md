# Metaphor Detector

Code for the Paper `Metaphor Detection for Low Resource Languages: From Zero-Shot to Few-Shot Learning in Middle High German`

The experiments lie in the experiments folder. Currently the data is still missing, we will provide the data (or instructions on how to obtain it) soon.
For now you can see how to use the programs by looking at the `reproduce` scripts in the expeirments folder. As soon as the data is available, you can just run the scripts to reproduce the experiments.

A few special mentions:
* `SimilarityNN.py` contains the definition of the feedforward network
* `train_network_from_texts.py` contains the basic unsupervised training of the network
* `finetune_network_on_datasets.py` contains the finetuning of the network

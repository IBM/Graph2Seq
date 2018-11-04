This is the code for the paper "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks".

To train your graph-to-sequence model, you need:

(1) Prepare your train/dev/test data which the form of:

    each line is a json object whose keys are "seq", "g_ids", "g_id_features", "g_adj":
    "seq" is a text which is supposed to be the output of the decoder
    "g_ids" is a mapping from the node ID to its ID in the graph
    "g_id_features" is a mapping from the node ID to its text features
    "g_adj" is a mapping from the node ID to its adjacent nodes (represented as thier IDs)

    See data/no_cycle/train.data as examples.


(2) Modify some hyper-parameters according to your task in the main/configure.py

(3) train the model by running the following code
    "train -sample_size_per_layer=xxx -sample_layer_size=yyy"
    The model that performs the best on the dev data will be saved in the dir "saved_model"

(4) test the model by running the following code
    "test -sample_size_per_layer=xxx -sample_layer_size=yyy"
    The prediction result will be saved in saved_model/prediction.txt






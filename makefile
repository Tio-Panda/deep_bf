WORKSPACE := /mnt/workspace/sgutierrezm/deep_bf
SERVER_HOME := /home/sgutierrezm/deep_bf
SERVER := sgutierrezm@ih-condor.ing.puc.cl

LOCAL := /home/panda/code/usm/deep_bf
DATASET_LOCAL := /home/panda/rf_data/dataset

CODE_TARGET := $(SERVER):$(SERVER_HOME)

PHONY: send_dataset, send_code, get_models

send_dataset:
	rsync -vrt $(DATASET_LOCAL)/webdataset $(SERVER):$(WORKSPACE)/dataset
	rsync -vrt $(DATASET_LOCAL)/samples_idx $(SERVER):$(WORKSPACE)/dataset

send_code:
	rsync -vrt $(LOCAL)/deep_bf $(CODE_TARGET)
	rsync -vrt $(LOCAL)/train.py $(CODE_TARGET)
	rsync -vrt $(LOCAL)/batchfile_single.sh $(CODE_TARGET)

get_models:
	rsync -azP $(SERVER):$(SERVER_HOME)/best_model.pth $(LOCAL)/best_model_server.pth

WORKSPACE := /mnt/workspace/sgutierrezm/deep_bf
SERVER_HOME := /home/sgutierrezm/deep_bf
SERVER := sgutierrezm@ih-condor.ing.puc.cl

LOCAL := /home/panda/code/usm/deep_bf
DATASET_LOCAL := /home/panda/rf_data/dataset

CODE_TARGET := $(SERVER):$(SERVER_HOME)

PHONY: send_dataset, send_code, get_models, migrate_db

train:
	python3 train.py -location local -e_id 0 -db_mode no-general

send_dataset:
	rsync -vrt $(DATASET_LOCAL)/webdataset_beamformer $(SERVER):$(WORKSPACE)/dataset

send_code:
	rsync -vrt $(LOCAL)/deep_bf $(CODE_TARGET)
	rsync -vrt $(LOCAL)/train.py $(CODE_TARGET)
	rsync -vrt $(LOCAL)/batchfile_single.sh $(CODE_TARGET)
	rsync -vrt $(LOCAL)/batchfile_all.sh $(CODE_TARGET)
	rsync -vrt $(LOCAL)/params.txt $(CODE_TARGET)
	rsync -vrt $(LOCAL)/requirements.txt $(CODE_TARGET)

get_models:
	rsync -azP $(SERVER):$(WORKSPACE)/models $(LOCAL)

migrate_db:
	rm -rf /home/panda/code/usm/deep_bf/deep_bf/config_registery/db/config_registery.db
	python3 -m deep_bf.config_registery.db.migrate

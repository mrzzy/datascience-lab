#
# makefile
# Makefile for Machine Learning
#

# Config Constants
PROJECT_DIRS := models data src

TRAIN := python3
MODEL_SRC := src/model.py
MODEL_PRODUCT := ./model.h5

GEN_DATA := python3
DATA_SRC := src/dataset.py
DATA_PRODUCT := data/dataset.pickle

# Utilities
FIND_MODEL_NO = $(shell ls models | sort -rn| head -1 | sed -ne "s/model_\([0-9]*\)\..*/\1/p")
NEXT_MODEL_NO = $(if $(FIND_MODEL_NO),$(shell expr $(FIND_MODEL_NO) + 1),1)

.DEFAULT: train
.PHONY: vcs setup train data

# Train target: Produce new version of model
train: $(MODEL_PRODUCT)
	mv $< models/model_$(NEXT_MODEL_NO).h5
	
$(MODEL_PRODUCT): $(MODEL_SRC) $(DATA_PRODUCT)
	$(info Training Model V$(NEXT_MODEL_NO)... )
	$(TRAIN) $<

# Setup target: setups project directory for machine learning
setup: $(PROJECT_DIRS) .gitignore

$(PROJECT_DIRS):
	mkdir -p $(PROJECT_DIRS)

.gitignore:
	@printf "data/ \nmodels/\n" > .gitignore

# Data target: regenerate dataset
data: $(DATA_PRODUCT)

$(DATA_PRODUCT): $(DATA_SRC)
	$(GEN_DATA) $<

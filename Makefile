SRC_DIR = $(PWD)
DIST_SERVER = $(SYNC_SERVER_NAME)
DIST_DIR = $(SYNC_DIST_DIR)

.PHONY: sync
sync:
	rsync -arzvc $(SRC_DIR) --delete --exclude-from $(PWD)/.rsync_exclude $(DIST_SERVER):$(DIST_DIR)

.PHONY: pull_runs
pull_runs:
	rsync -arzvc --exclude model.pt $(DIST_SERVER):$(DIST_DIR)/debias-BERT/runs $(SRC_DIR)

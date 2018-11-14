
# set log dir below
echo '\nexport LOG_DIR=/mnt/storage/log-experiments' >> ~/.bashrc
source ~/.bashrc

# create directories
mkdir experiments
mkdir ${LOG_DIR}/logs
mkdir ${LOG_DIR}/results
mkdir ${LOG_DIR}/runs
mkdir ${LOG_DIR}/weights

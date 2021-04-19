#!/bin/bash
set -e
set -x

CFG=$1
nEPOCH=$2
FEAT_LIST=$3 # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=${4:-8}

EPOCH=100
echo "Testing all checkpoints."

while [ $nEPOCH -gt $EPOCH ]
do
	EPOCH=$(( EPOCH+10 ))
	bash benchmarks/dist_test_svm_epoch.sh $CFG $EPOCH $FEAT_LIST $GPUS
	echo "Evaluation complete for epoch: "
	echo "$EPOCH"
	
done

#!/bin/bash

export PYTHONPATH=$(pwd)/workspace/src
TESTDIR=$(pwd)/workspace/tests
TRIAL_NUM=100

python run.py  --submit-dir $(pwd)/workspace --test-dir ${TESTDIR} --parallel 1 --trial-number ${TRIAL_NUM} --time-out 3600
python evaluate.py --test-dir ${TESTDIR} --trial-number ${TRIAL_NUM}

rm -r workspace/tests/**/stderr.txt
rm -r workspace/tests/**/stdout.txt
rm -rf workspace/tests/**/results workspace/tests/**/work
rm num_file_exists.json

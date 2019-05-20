#!/bin/sh

python train.py --model_path ../models/males --train_csv csvfiles/males/train.csv --val_csv csvfiles/males/validate.csv
python train.py --model_path ../models/females --train_csv csvfiles/females/train.csv --val_csv csvfiles/females/validate.csv
python train.py --model_path ../models/random --train_csv csvfiles/random/train.csv --val_csv csvfiles/random/validate.csv

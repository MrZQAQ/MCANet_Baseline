#!/usr/bin/ bash

conda activate MolTrans

python train.py --task DrugBank --epochs 200 --batch-size 16 2>&1 | tee output_DrugBank.txt

python train.py --task KIBA --epochs 200 --batch-size 16 2>&1 | tee output_KIBA.txt

python train.py --task Davis --epochs 200 --batch-size 16 2>&1 | tee output_Davis.txt
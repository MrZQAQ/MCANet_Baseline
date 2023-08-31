#! /bin/bash

echo "DeepConvDTI K-Fold 5 scripts"
echo "Copyright © 2022 MrZQAQ. All rights reserved."

TF_VERSION=`pip list | grep tensorflow | awk '{print $2}' | awk -F '.' '{print $1}'`
KERAS_VERSION=`pip list | grep keras | awk '{print $2}'`
H5PY_VERSION=`pip list | grep h5py | awk '{print $2}' | awk -F '.' '{print $1}'`

# tf version 1 and h5py version 2 is needed
if [ $TF_VERSION != 1 ]
then
    echo "Tensorflow must be version 1.x"
    exit 1
fi
if [ $H5PY_VERSION == 3 ]
then
    echo "h5py package should downgrade to 2.x"
    pip uninstall h5py -y &> /dev/null
    pip install h5py==2.10.0 &> /dev/null
fi

cd DeepConv-DTI

# create save dir
mkdir ./result
mkdir ./pridict
mkdir ./train
for dataset_name in Davis KIBA DrugBank
do
    mkdir ./result/${dataset_name}
    mkdir ./pridict/${dataset_name}
    mkdir ./train/${dataset_name}
done


for dataset_name in Davis KIBA DrugBank
do
    for fold in 1 2 3 4 5
    do
        # train model
        python DeepConvDTI.py ./data/${dataset_name}/train/${fold}/dti.csv ./data/${dataset_name}/train/${fold}/drug.csv ./data/${dataset_name}/train/${fold}/protein.csv \
                                --validation -n validation_dataset \
                                -i ./data/${dataset_name}/validation/${fold}/dti.csv \
                                -d ./data/${dataset_name}/validation/${fold}/drug.csv \
                                -t ./data/${dataset_name}/validation/${fold}/protein.csv \
                                -b 16 -o ./train/${dataset_name}/k${fold}.csv -m ./train/${dataset_name}/k${fold}.model \
                                -e 50
        # pridict
        python predict_with_model.py ./train/${dataset_name}/k${fold}.model \
                --test-name ${dataset_name} --test-dti-dir ./data/${dataset_name}/test/dti.csv --test-drug-dir ./data/${dataset_name}/test/drug.csv --test-protein-dir ./data/${dataset_name}/test/protein.csv --with-label \
                --output ./pridict/${dataset_name}/k${fold}.csv
        # evaluate
        python evaluate_performance.py ./pridict/${dataset_name}/k${fold}.csv --test-name ${dataset_name} --no-threshold --evaluation-output ./result/${dataset_name}/k${fold}.csv
    done
done

# MMKG

## pretrain data
Link: https://drive.google.com/drive/folders/19FWEi5v0Ds3rAxQHfOfRq1nhE1beaZjL?hl=vi

then create a folder name pre_train and put all data in.


## Step 1: 
cd code
python process_datasets.py

## Step 2: run code

- WN9-IMG: 

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset WN9IMG --model model_wn --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-3 --max_epochs 100 \
--valid 5 -train -id 0 -save -weight

- FB-IMG:

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset FB9IMG --model model_fb --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 5000 --regularizer N3 --reg 1e-3 --max_epochs 150 \
--valid 5 -train -id 0 -save -weight

- DB15k:

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset DB15k --model model_db --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 5000 --regularizer N3 --reg 3e-3 --max_epochs 150 \
--valid 5 -train -id 0 -save -weight

- MKG-W:

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset MKG-W --model model_mkgw --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-3 --max_epochs 150 \
--valid 5 -train -id 0 -save -weight

- MKG-Y:

CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset MKG-Y --model model_mkgy --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-3 --max_epochs 150 \
--valid 5 -train -id 0 -save -weight

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#apt update && apt install -y libopenmpi-dev && apt install -y build-essential
yum install -y openmpi-devel && export CC=/usr/lib64/openmpi/bin/mpicc
pip install -r ./code/requirements.txt 
rm -r ~/.cache/torch_extensions/utils

HIDDEN_SIZE_BEGIN=2096
# HIDDEN_SIZE_BEGIN=2400

for addition in 0 1 2 3 4 5 6 7 8 9 10
do
    HIDDEN_SIZE=`expr ${HIDDEN_SIZE_BEGIN} + ${addition} \* 32`
    echo "current hidden size is ${HIDDEN_SIZE} ..."

    CXX=g++ deepspeed --num_gpus=2 ./code/train.py \
    # CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ./code/train.py \
    --deepspeed ./code/ds_auto_config.json \
    --lr=1e-4 \
    --epochs=10 \
    --train_batch_size=2 \
    --seq_length=512 \
    --max_predictions_per_seq=80 \
    --num_hidden_layers=24 \
    --num_attention_heads=16 \
    --hidden_size=${HIDDEN_SIZE} \
    --vocab_size=30522 \
    --dataset_size=2

    if [ $? != 0 ]
    then
        FINAL_HIDDEN_SIZE=`expr ${HIDDEN_SIZE} - ${addition} \* 32`
        echo "the final hidden size is ${FINAL_HIDDEN_SIZE} !!!!"
        break
    fi
done

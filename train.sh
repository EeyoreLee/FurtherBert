pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
apt install -y libopenmpi-dev && \
pip install -r ./code/requirements.txt && \
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 ./code/train.py \
--deepspeed ./code/ds_auto_config.json \
--lr=1e-4 \
--epochs=10 \
--train_batch_size=2 \
--seq_length=512 \
--max_predictions_per_seq=80 \
--num_hidden_layers=24 \
--num_attention_heads=16 \
--hidden_size=2256 \
--vocab_size=30522 \
--dataset_size=2 && \
python ./code/convert_fp32_to_bert_large.py
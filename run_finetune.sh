for lr in 5e-4
do
echo ${lr}
python learn.py \
    --dataset="analogy" \
    --model="Analogy" \
    --batch_size=2000 \
    --learning_rate=${lr} \
    --max_epochs=500 \
    --finetune \
    --ckpt="checkpoint"
done
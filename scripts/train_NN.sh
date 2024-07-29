export CUDA_VISIBLE_DEVICES=1,3
python train.py \
    --model_type "NN" \
    --data_filename "X_5120_L8T24_excess.pkl"\
    --labels_filename "y_5120_L8T24_excess.pkl"\
    --batch_size 1024 \
    --train_bystep \
    --train_step 5000 \
    --eval_step 200 \
    --inference \
    --infeasibility_loss



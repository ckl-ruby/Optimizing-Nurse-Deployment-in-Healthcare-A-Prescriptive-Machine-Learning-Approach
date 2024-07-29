python train.py \
    --model_type "rdforest" \
    --data_filename "X_5120_L8T24.pkl"\
    --labels_filename "y_5120_L8T24.pkl"\
    --n_estimator 3 \
    --inference \
    # --inference_from_ckp \
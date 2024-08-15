export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "NN" \
    --checkpoint_path "./checkpoint/exnt_real_calculated_obj"\
    --data_filename "EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl"\
    --learning_rate 5e-3 \
    --g_type "quadratic" \
    --round_threshold 0.5 \
    --batch_size 768 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 768 \
    --mid_layer 5 \
    --loss_combo "1000100"\
    --loss_lambda "1" "1" "1" "1" "1" "1" "1"\
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --save_name_suffix "" \
    --optimizer "adam" \
    --inference_step 500 \
    --exnt \
    --calculated_obj \
    # --inference_only \



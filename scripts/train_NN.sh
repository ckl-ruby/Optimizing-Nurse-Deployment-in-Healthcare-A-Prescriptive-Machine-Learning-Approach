export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "NN" \
    --checkpoint_path "./checkpoint/qfrac"\
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl"\
    --learning_rate 5e-3 \
    --g_type "quadratic" \
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
    --inference_step 100 \
    # --inference_only 

#    --inference_model_path "./checkpoint/train_inference/selected/NN_on_10240_L8_excess_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue.pkl_step2000_bs512_hd768_ly5_loss100010_lambda['1', '1', '1', '1', '1', '1']_adam_lr0.005"\



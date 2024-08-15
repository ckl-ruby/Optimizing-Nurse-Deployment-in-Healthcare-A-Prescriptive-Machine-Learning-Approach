export CUDA_VISIBLE_DEVICES=5
python train.py \
    --model_type "stage2_NN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/qfrac"\
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl"\
    --stage1_model_path "./checkpoint/qfrac/NN_on_10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl_step2000_bs768_hd768_ly5_loss1000100_lambda['1', '1', '1', '1', '1', '1', '1']_adam_lr0.005_round0.5/best_NN_cost_obj.pth" \
    --batch_size 768 \
    --round_threshold 0.5 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 768 \
    --mid_layer 5 \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --inference_step 500\
    # --inference_only \


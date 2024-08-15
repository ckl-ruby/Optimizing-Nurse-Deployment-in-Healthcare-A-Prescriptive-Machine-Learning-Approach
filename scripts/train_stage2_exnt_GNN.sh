export CUDA_VISIBLE_DEVICES=1
python train.py \
    --model_type "stage2_GNN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/exnt_real_calculated_obj"\
    --data_filename "EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl"\
    --stage1_model_path "./checkpoint/exnt_real_calculated_obj/GNN_on_EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl_step2000_bs1024_hd1024_ly5_loss100010_lambda['1', '1', '1', '1', '1', '1']_adam_lr0.005_round0.5/best_GNN_cost_obj.pth" \
    --batch_size 1024 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 1024 \
    --mid_layer 5 \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --inference_step 500 \
    --exnt \
    # --inference_only \


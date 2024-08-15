export CUDA_VISIBLE_DEVICES=4
python train.py \
    --model_type "stage2_GNN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/qfrac"\
    --data_filename "10240_L5_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl"\
    --stage1_model_path "./checkpoint/qfrac/GNN_on_10240_L5_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl_step2000_bs1024_hd1024_ly5_loss1000100_lambda['1', '1', '1', '1', '1', '1', '1']_adam_lr0.005_round0.5/best_GNN_cost_obj.pth" \
    --batch_size 1024 \
    --train_bystep \
    --train_step 10000 \
    --eval_step 400 \
    --inference \
    --d_model 1024 \
    --mid_layer 5 \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --inference_step 1000 \
    # --inference_only \


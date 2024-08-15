export CUDA_VISIBLE_DEVICES=2
python train.py \
    --model_type "GNN" \
    --checkpoint_path "./checkpoint/qfrac"\
    --learning_rate 5e-3 \
    --g_type "quadratic" \
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl"\
    --batch_size 1024 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 100 \
    --inference \
    --d_model 1024 \
    --mid_layer 5 \
    --loss_combo "1000100"\
    --loss_lambda "1" "1" "1" "1" "1" "1" "1"\
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --save_name_suffix "" \
    --optimizer "adam" \
    --inference_step 100 \
    # --inference_only 
    # --inference_model_path "./checkpoint/train_inference/selected/GNN_on_10240_L20_excess_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue.pkl_step1000_bs1024_hd1024_ly5_loss100010_lambda['1', '1', '1', '1', '1', '1']_adam_lr0.005"\
# 



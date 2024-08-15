# Setup environment
```
cd scheduling
pip install -r requirements.txt
```
# Data generation
For data generation, we use two solver, COPT and Gurobi. Please make sure you have access to those solver when running the generation script.

## Ex post dataset
```bash
python data_generator.py \
    --num_location 8 \
    --num_samples 10240 \
    --saving_path "./data" \
    --zero_threshold 1e-5 \
    --loss_type "quadratic" \ 
    --Kxi_bound "10" "200"\
    --D_bound "0" "2"\
    --with_quard_farc \
    --with_int_solution \
```
This script will generate a dataset with 8 locations, quadratic objective (with parameters ax^2 + bx + c), all location demand will bounded between 10 and 200, all transfer cost D will be bounded between 0 to 2. The solver will provide a float and an Int solution.

## Ex ante dataset
```bash
python data_generator.py \
    --num_location 8 \
    --num_samples 10240 \
    --saving_path "./data" \
    --zero_threshold 1e-5 \
    --loss_type "quadratic" \
    --Kxi_bound "10" "200"\
    --D_bound "0" "2"\
    --exnt_sample_amount 100 \
    --with_int_solution \
    --exnt \
```

This script will generate a ex ante dataset, quadratic objective (x^2), all location demand will bounded between 10 and 200, all transfer cost D will be bounded between 0 to 2. The solver will provide a float and an Int solution. Specifically, the solver will sample 100 xi to approximate the normal distribution for each location demands.


# Training & evaluation

## Ex poss
Ex pose training will include ANN and GNN parts.

Stage One
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "NN" \
    --checkpoint_path "./checkpoint"\
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
    --loss_combo "100010"\
    --loss_lambda "1" "1" "1" "1" "1" "1" \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --save_name_suffix "" \
    --optimizer "adam" \
    --inference_step 100 \
    # --inference_only 
```
This script will train a ANN network to minimize the `quadratic` objective function using the dataset at `data_filename`. It will train for 2000 batches and each 400 batches will enter the evaluation loop and each evaluation loop will go through `inference_step` datas from the evaluation dataset. The best model evaluated by the list of `model_selection_strategy`  will be saved. Specifically, we can customize the loss combination by `loss_combo` and the weights of each loss by `loss_lambda`. Please check the `hybird_loss` in `Utils.py` for more details. If you finish training, use `inference_only` to skip the training. Replace `model_type` to `GNN` for Graph neural network training.

Stage two
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "stage2_NN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/qfrac"\
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl"\
    --stage1_model_path "./checkpoint/{your_checkpoint}.pth" \
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
```

After stage 1 training, we will refine the output by stage 2 network. Put the model path on `stage1_model_path` and set `model_type` to `stage2_NN` we can start the stage2 training. The training will load the Int solution from the dataset, please make sure the dataset on `data_filename` contains integer solution.


## En ante
Stage 1

```bash
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
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --save_name_suffix "" \
    --optimizer "adam" \
    --inference_step 500 \
    --exnt \
    --calculated_obj \
    # --inference_only \
```
Similar to ex pose, we only need to add `exnt` and replace the dataet with the EX ante dataset. If we add `calculated_obj`, both label and network output objective value will calculated by a known formula, otherwise, we will use the objective value provided by the solver.

Stage 2

```bash
export CUDA_VISIBLE_DEVICES=4
python train.py \
    --model_type "stage2_NN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/exnt_real_calculated_obj"\
    --data_filename "EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl"\
    --stage1_model_path "./checkpoint/{your_checkpoint}.pth" \
    --batch_size 768 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 768 \
    --mid_layer 5 \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --inference_step 500 \
    --exnt \
    # --inference_only \
```

Similarly, add the `stage1_model_path` and then the training is ready to run.
# Optimizing Nurse Deployment in Healthcare: A Prescriptive Machine Learning Approach

This repository contains the code and data used for the paper **"Optimizing Nurse Deployment in Healthcare: A Prescriptive Machine Learning Approach"**. The paper introduces a novel approach to optimizing nurse deployment using machine learning models, specifically focusing on neural networks and graph neural networks, combined with mixed-integer programming solvers. The goal is to enhance decision-making in healthcare settings by providing optimized deployment strategies.

## Table of Contents

- [Setup Environment](#setup-environment)
- [Data Generation](#data-generation)
  - [Ex post Dataset](#ex-post-dataset)
  - [Ex ante Dataset](#ex-ante-dataset)
- [Training & Evaluation](#training--evaluation)
  - [Ex post Training](#ex-post-training)
    - [Stage One](#stage-one)
    - [Stage Two](#stage-two)
  - [Ex ante Training](#ex-ante-training)
    - [Stage One](#stage-one-1)
    - [Stage Two](#stage-two-1)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Setup Environment

To begin, ensure that you have the necessary dependencies installed. You can do this by navigating to the `scheduling` directory and installing the required packages listed in the `requirements.txt` file:

```bash
cd scheduling
pip install -r requirements.txt
```

Please note that this project requires access to **COPT** and **Gurobi** solvers for data generation and optimization tasks. Ensure that these solvers are installed and properly configured on your system.

## Data Generation

Data generation is a critical step in our approach. We provide scripts to generate both **Ex post** and **Ex ante** datasets using the solvers mentioned above. The datasets are used to train and evaluate the neural network models described in the paper.

### Ex post Dataset

To generate the Ex post dataset, run the following script:

```bash
python data_generator.py \
    --num_location 8 \
    --num_samples 10240 \
    --saving_path "./data" \
    --zero_threshold 1e-5 \
    --loss_type "quadratic" \
    --Kxi_bound "10" "200" \
    --D_bound "0" "2" \
    --with_quard_farc \
    --with_int_solution
```

This script generates a dataset with:

- **8 locations**
- A **quadratic objective function** (`ax^2 + bx + c`)
- **Location demands** bounded between **10** and **200**
- **Transfer costs (D)** bounded between **0** and **2**
- Both **floating-point** and **integer solutions** provided by the solver

### Ex ante Dataset

For generating the Ex ante dataset, use the following command:

```bash
python data_generator.py \
    --num_location 8 \
    --num_samples 10240 \
    --saving_path "./data" \
    --zero_threshold 1e-5 \
    --loss_type "quadratic" \
    --Kxi_bound "10" "200" \
    --D_bound "0" "2" \
    --exnt_sample_amount 100 \
    --with_int_solution \
    --exnt
```

This script generates an Ex ante dataset with similar parameters as the Ex post dataset. However, it includes a step where **100 samples** are drawn to approximate the normal distribution of each location's demands.

## Training & Evaluation

The training and evaluation process is divided into two stages: **Stage One** for initial model training and **Stage Two** for refining the model output.

### Ex post Training

#### Stage One

To train a neural network (ANN or GNN) on the Ex post dataset, use the following script:

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "NN" \
    --checkpoint_path "./checkpoint" \
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl" \
    --learning_rate 5e-3 \
    --g_type "quadratic" \
    --batch_size 768 \
    --train_bystep \
    --train_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 768 \
    --mid_layer 5 \
    --loss_combo "100010" \
    --loss_lambda "1" "1" "1" "1" "1" "1" \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --save_name_suffix "" \
    --optimizer "adam" \
    --inference_step 100
    # --inference_only 
```

**Explanation of Parameters:**

- `--model_type "NN"`: Specifies the use of a Neural Network. Replace with `"GNN"` for Graph Neural Network training.
- `--checkpoint_path "./checkpoint"`: Directory to save model checkpoints.
- `--data_filename "... .pkl"`: Path to the dataset file.
- `--learning_rate 5e-3`: Learning rate for the optimizer.
- `--batch_size 768`: Number of samples per batch.
- `--train_bystep`: Enables training by steps.
- `--train_step 2000`: Total number of training steps.
- `--eval_step 400`: Evaluation occurs every 400 steps.
- `--inference`: Enables inference during evaluation.
- `--d_model 768`: Dimensionality of the model.
- `--mid_layer 5`: Number of middle layers in the network.
- `--loss_combo "100010"`: Specifies the combination of losses to use. Refer to `hybrid_loss` in `Utils.py` for details.
- `--loss_lambda "1" "1" "1" "1" "1" "1"`: Weights for each loss component.
- `--model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj"`: Metrics for selecting the best model.
- `--inference_step 100`: Number of inference steps during evaluation.
- `# --inference_only`: Uncomment to skip training and perform inference only.

#### Stage Two

After Stage One, the output can be refined in Stage Two:

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "stage2_NN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/qfrac" \
    --data_filename "10240_L8_quadratic_kxi['10', '200']_D['0', '2']_zerosFalse_intsolutionTrue_qfracTrue.pkl" \
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
    --inference_step 500
    # --inference_only
```

**Key Points:**

- `--model_type "stage2_NN"`: Specifies Stage Two Neural Network training.
- `--stage1_model_path "./checkpoint/{your_checkpoint}.pth"`: Path to the Stage One model checkpoint.
- `--round_threshold 0.5`: Threshold for rounding during post-processing.
- Ensure the dataset contains integer solutions for effective refinement.

### Ex ante Training

#### Stage One

For Ex ante training, use the following script:

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_type "NN" \
    --checkpoint_path "./checkpoint/exnt_real_calculated_obj" \
    --data_filename "EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl" \
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
    --calculated_obj
    # --inference_only
```

**Additional Flags:**

- `--exnt`: Indicates that the dataset is Ex ante.
- `--calculated_obj`: Uses a known formula to calculate objective values. If omitted, objective values provided by the solver are used.

#### Stage Two

Finally, refine the Ex ante model in Stage Two:

```bash
export CUDA_VISIBLE_DEVICES=4
python train.py \
    --model_type "stage2_NN" \
    --g_type "quadratic" \
    --checkpoint_path "./checkpoint/exnt_real_calculated_obj" \
    --data_filename "EXNT_10240_L8_quadratic_D['0', '2']_zerosFalse_intsolutionTrue_m100.pkl" \
    --stage1_model_path "./checkpoint/{your_checkpoint}.pth" \
    --batch_size 768 \
    --train_bystep \
    --train

_step 2000 \
    --eval_step 400 \
    --inference \
    --d_model 768 \
    --mid_layer 5 \
    --model_selection_strategy "eval_loss" "f1" "infesibility" "cost_obj" \
    --inference_step 500 \
    --exnt
    # --inference_only
```

Ensure to set the correct `--stage1_model_path` and include the `--exnt` flag to indicate the use of Ex ante datasets.

## Conclusion

This repository provides all necessary scripts and instructions to reproduce the results presented in our paper. By following the steps outlined above, you should be able to generate data, train models, and evaluate their performance. Please ensure you have the required solvers (**COPT** and **Gurobi**) installed to avoid any issues during data generation or optimization tasks.

We hope this repository helps you explore the potential of prescriptive machine learning in healthcare settings, specifically in optimizing nurse deployment strategies.

## Contact

For any questions, issues, or contributions, please feel free to open an issue or contact us directly.
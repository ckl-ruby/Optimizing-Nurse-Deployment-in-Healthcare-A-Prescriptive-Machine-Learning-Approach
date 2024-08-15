python data_generator.py \
    --num_location 8 \
    --num_samples 10240 \
    --saving_path "./data" \
    --zero_threshold 1e-5 \
    --loss_type "quadratic" \
    --Kxi_bound "10" "200"\
    --D_bound "0" "2"\
    --exnt_sample_amount 100 \
    --with_quard_farc \
    --with_int_solution \
    # --exnt \
    # --keep_zero_solution
    
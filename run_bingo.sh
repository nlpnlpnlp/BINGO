
aspect=0
sparsity_percentage=0.1
optimizer=BINGO
data_type='beer'
epochs=400
gpu=0
cls_lambda=1.0 
sparsity_lambda=1.0 
continuity_lambda=1.0


log_dir=./new_log/$data_type/'spa'$sparsity_percentage/'as'$aspect/
mkdir -p $log_dir
python -u main_bingo.py --hidden_dim 200 --save 0 --dropout 0.2 --lr 0.0001 \
        --data_type beer --batch_size 128 --gpu $gpu --sparsity_percentage $sparsity_percentage \
        --cls_lambda $cls_lambda --sparsity_lambda $sparsity_lambda --continuity_lambda $continuity_lambda --epochs $epochs --aspect 0 \
        --optimizer $optimizer \
        --results_dir $log_dir > $log_dir/cmd1_$sparsity_percentage.log


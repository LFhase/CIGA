# DrugOOD-assay
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_assay' --seed '[1,2,3,4,5]' --dropout 0.5 --contrast 8 -c_in 'raw'  -c_rep 'feat'  --spu_coe 0
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_assay' --seed '[1,2,3,4,5]' --dropout 0.5 --contrast 1 -c_in 'raw'  -c_rep 'feat'  --spu_coe 1
# DrugOOD-scaffold
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5 --contrast 32 -c_in 'feat'  -c_rep 'feat' -s_rep 'conv'  --spu_coe 0
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5 --contrast 16 -c_in 'feat'  -c_rep 'feat' -s_rep 'conv'  --spu_coe 1
# DrugOOD-size
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_size' --seed '[1,2,3,4,5]' --dropout 0.1 --contrast 16 -c_in 'feat'  -c_rep 'feat'  --spu_coe 0
python main.py --eval_metric 'auc' --r 0.8 --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ic50_size' --seed '[1,2,3,4,5]' --dropout 0.1 --contrast 2  -c_in 'feat'  -c_rep 'feat'  --spu_coe 1

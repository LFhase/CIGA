# nci1
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'nci1' --r 0.6 --contrast 0.5 --spu_coe 0 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'nci1' --r 0.6 --contrast 1   --spu_coe 1 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 
# nci109
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'nci109' --r 0.7 --contrast 2 --spu_coe 0 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'nci109' --r 0.7 --contrast 2 --spu_coe 1 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 
# proteins
python main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'proteins' --r 0.3 --contrast 0.5 --spu_coe 0 --model 'gin' --pooling 'max' --dropout 0.3 --eval_metric 'mat' 
python main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'proteins' --r 0.3 --contrast 0.5 --spu_coe 1 --model 'gin' --pooling 'max' --dropout 0.3 --eval_metric 'mat' 
# DD
python main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'dd' --r 0.3 --contrast 2 --spu_coe 0 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 
python main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'dd' --r 0.3 --contrast 2 --spu_coe 1 --model 'gcn' --dropout 0.3 --eval_metric 'mat' 

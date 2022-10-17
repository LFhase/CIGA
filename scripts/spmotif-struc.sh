# spmotif-struc b=0.33
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.33 --r 0.25 --contrast 8 --spu_coe 0 --model 'gcn' --dropout 0.  
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.33 --r 0.25 --contrast 8 --spu_coe 1 --model 'gcn' --dropout 0. 
# spmotif-struc b=0.60
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.60 --r 0.25 --contrast 0.5 --spu_coe 0 --model 'gcn' --dropout 0.  
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.60 --r 0.25 --contrast 4   --spu_coe 2 --model 'gcn' --dropout 0.  
# spmotif-struc b=0.90
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.90 --r 0.25 --contrast 4 --spu_coe 0 --model 'gcn' --dropout 0. 
python main.py  -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5]' --num_layers 3 --dataset 'SPMotif' --bias 0.90 --r 0.25 --contrast 4 --spu_coe 2 --model 'gcn' --dropout 0. 

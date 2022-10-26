# CMNIST-sp
python main.py --r 0.8 --num_layers 3  --batch_size 32 --emb_dim 32 --model 'gcn' -c_dim 128 --dataset 'CMNIST' --seed '[1,2,3,4,5]' --contrast 32 --spu_coe 0 -c_in 'raw'  -c_rep 'feat'
python main.py --r 0.8 --num_layers 3  --batch_size 32 --emb_dim 32 --model 'gcn' -c_dim 128 --dataset 'CMNIST' --seed '[1,2,3,4,5]' --contrast 16 --spu_coe 16 -c_in 'raw'  -c_rep 'feat'


# Graph-SST5
python main.py --r 0.5 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-SST5' --seed '[1,2,3,4,5]' --contrast 8 --spu_coe 0 -c_in 'raw'  -c_rep 'feat'
python main.py --r 0.5 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-SST5' --seed '[1,2,3,4,5]' --contrast 4 --spu_coe 1 -c_in 'raw'  -c_rep 'feat'

# Graph-Twitter
python main.py --r 0.6 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-Twitter' --seed '[1,2,3,4,5]' --contrast 8  --spu_coe 0 -c_in 'feat'  -c_rep 'feat' 
python main.py --r 0.6 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-Twitter' --seed '[1,2,3,4,5]' --contrast 8  --spu_coe 0.5 -c_in 'feat'  -c_rep 'feat'




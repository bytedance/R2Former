# train reranker with fixed (default) backbone and global mining
python3 -u train_reranker.py --dataset_name=msls --backbone deit --aggregation=gem --mining global --neg_hardness 100 --datasets_folder ../datasets_vg/datasets  --save_dir rerank --lr 0.0005 --cos --fc_output_dim 256 --num_workers 8 --warmup 5 --optim adamw --epochs_num 50 --patience 25 --negs_num_per_query 1 --queries_per_epoch 50000 --cache_refresh_rate 10000 --train_batch_size 64 --infer_batch_size 256 --rerank_batch_size 20 --save_best 0

# finetune global retrieval and reranking modules together, use --fix 0 to enable training of global retrieval module, add --finetune 1 may have better result
python3 -u train_reranker.py --fix 0 --dataset_name=msls --backbone deit --aggregation=gem --mining partial --datasets_folder ../datasets_vg/datasets  --save_dir finetune --resume log/rerank/best_model.pth --lr 0.00001 --cos --fc_output_dim 256 --num_workers 8 --warmup 5 --optim adamw --epochs_num 20 --patience 50 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000 --train_batch_size 48 --infer_batch_size 256 --rerank_batch_size 20 --save_best 0

# finetune on Pitts30k for urban datasets
# python3 -u train_reranker.py --fix 0 --dataset_name=pitts30k --backbone deit --aggregation=gem --mining full --datasets_folder ../datasets_vg/datasets  --save_dir pitts30k_finetune --resume CVPR23_DeitS_Rerank.pth  --lr 0.00001 --fc_output_dim 256  --cos --warmup 5 --optim adamw --epochs_num 50 --patience 10 --negs_num_per_query 1 --queries_per_epoch 5000 --cache_refresh_rate 1000 --train_batch_size 16 --infer_batch_size 256 --rerank_batch_size 4  

# use backbone="resnet50" to train reranker with resnet backbone
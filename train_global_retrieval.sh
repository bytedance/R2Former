# Train a ViT-Small global retrieval model, initialized with DeiT-Small weights from ImageNet-1K
python3 -u train_deit.py --dataset_name msls --backbone deit --aggregation gem --mining partial --datasets_folder ../datasets_vg/datasets --save_dir global_retrieval --lr 0.00001 --fc_output_dim 256 --train_batch_size 16 --infer_batch_size 256 --num_workers 8 --epochs_num 100 --patience 10 --negs_num_per_query 2 --queries_per_epoch 50000 --cache_refresh_rate 10000

# Use backbone="resnet50conv5" to train a resnet model with VG benchmark
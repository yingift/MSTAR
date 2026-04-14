
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=10021 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/siglip/word/stage1_pt_siglip512.yaml
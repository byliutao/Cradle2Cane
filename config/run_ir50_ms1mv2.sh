python main.py \
    --data_root ../dataset \
    --train_data_path faces_webface_112x112 \
    --val_data_path faces_webface_112x112 \
    --prefix ir50_ms1mv2_adaface_ep40 \
    --use_mxrecord \
    --gpus 2 \
    --arch ir_50 \
    --weight_decay 6e-4 \
    --batch_size 512 \
    --num_workers 16 \
    --epochs 40  \
    --lr_milestones 18,28,36 \
    --lr 0.1  \
    --head adaface \
    --m 0.4 \
    --h 0.25 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2 \
    --custom_num_class 15572 \
2>&1 | tee experiments/train_$(date +%Y%m%d_%H%M%S).log

# --resume_from_checkpoint experiments/ir50_ms1mv2_adaface_ep50_10-28_1/epoch=17-step=26046.ckpt \

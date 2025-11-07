DATE=$(date +%Y%m%d_%H%M%S)
NAME=ir50_ms1mv2_adaface_gpu9_${DATE}

python main.py \
    --data_root ../dataset \
    --train_data_path faces_webface_112x112 \
    --val_data_path faces_webface_112x112 \
    --prefix ${NAME} \
    --gpus 4 \
    --arch ir_101 \
    --weight_decay 6e-4 \
    --batch_size 1024 \
    --num_workers 16 \
    --epochs 70  \
    --lr_milestones 48,58,66 \
    --lr 0.4  \
    --head adaface \
    --m 0.3 \
    --h 0.25 \
    --low_res_augmentation_prob 0.1 \
    --crop_augmentation_prob 0.1 \
    --photometric_augmentation_prob 0.1 \
    --custom_num_class 13572 \
2>&1 | tee experiments/train_${NAME}.log

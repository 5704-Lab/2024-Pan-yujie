export MODEL_NAME="path/to/stable-diffusion-v1-5"
export INSTANCE_DIR="none"

export NAME="hazelnut"
export ANOMALY="hole"
export OUTPUT_DIR="all_generate/$NAME/$ANOMALY"

CUDA_VISIBLE_DEVICES={id} accelerate launch \
    --main_process_port=30005 \
    train_dreambooth_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="" \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=5000 \
    --resume_from_checkpoint "latest" \
    --mvtec_name=$NAME \
    --mvtec_anamaly_name=$ANOMALY \
    --rank 32 \
    --seed 32 \
    --train_text_encoder

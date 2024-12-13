export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="fine_tuned_rks_lora_new"
export HUB_MODEL_ID="rks-lora-diffusion"
export DATASET_NAME="AaditD/multilingual_rks"

accelerate launch --mixed_precision="bf16"  train_text_to_image_lora_safe.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="caption_alt_text_description" \
  --cache_dir="/opt/dlami/nvme" \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Denkmal für die Opfer des Zweiten Weltkrieges, Weinhübel" \
  --seed=1337
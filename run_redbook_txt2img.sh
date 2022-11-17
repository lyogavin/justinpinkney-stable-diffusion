#!/bin/bash

# ref: https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
# ref: https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda
# ref: https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning
# repo: https://github.com/lyogavin/justinpinkney-stable-diffusion

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $run_ts"

echo "START TIME: $(date)"

ROOT_DIR=/home/ubuntu/cloudfs/saved_models/stable_diffusion/finetune_redbook_txt2img/

if [ ! -d ${ROOT_DIR} ]; then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

LOGS_DIR=/home/ubuntu/cloudfs/saved_models/stable_diffusion/finetune_redbook_txt2img/logs

if [ ! -d ${LOGS_DIR} ]; then
  mkdir -p ${LOGS_DIR}
  echo ${LOGS_DIR} created!!!!!!!!!!!!!!
else
  echo ${LOGS_DIR} exist!!!!!!!!!!!!!!!
fi

config_yaml="$ROOT_DIR/base_config.yaml"

# dataset gen'd by ghostai_training/ocr_title_to_image/gen_data.ipynb
# from :https://github.com/invoke-ai/InvokeAI/blob/main/configs/stable-diffusion/v1-finetune.yaml
cat <<EOT >$config_yaml
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 8
    num_workers: 64
    #num_val_workers: 0 # Avoid a weird val dataloader issue
    train:
      target: ldm.data.simple.hf_dataset
      params:
        name: /home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/txt2image_dataset_sim_scored_filtered_1668610069
    validation:
      target: ldm.data.simple.TextOnly
      params:
        captions:
        - "Matcha cheese milk cover"
        - "The baby's favorite Friso Friso Milk Powder"
        - "Two dogs"
        - "YulinJewelry"
        output_size: 512
        n_gpus: 2 # small hack to make sure we see all our samples
lightning:
  modelcheckpoint:
    every_n_train_steps: 100
model:
  base_learning_rate: 5.0e-05
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.012
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: caption
    image_size: 32
    channels: 4
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        - 4
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 1280
        use_checkpoint: true
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.BERTEmbedder
      params:
        n_embed: 1280
        n_layer: 32

EOT

#BATCH_SIZE=4
N_GPUS=8
#ACCUMULATE_BATCHES=1
TRAIN_NAME=finetune_redbook_txt2img
gpu_list=0,1,2,3

# following: https://github.com/lyogavin/justinpinkney-stable-diffusion
#curl -L -x socks5h://localhost:8123 https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned/resolve/main/sd-clip-vit-l14-img-embed_ema_only.ckpt -o ../saved_models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt

ckpt_path=/home/ubuntu/cloudfs/saved_models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt

# testing setup
#    --every_n_train_steps 100 \
export CMD=" main.py \
    --train \
    --base $config_yaml \
    --gpus $gpu_list \
    --logdir LOGS_DIR \
    --name $TRAIN_NAME    --scale_lr True \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from $ckpt_path \
    --every_n_train_steps 100 \
    "

echo $CMD

python $CMD

#python main.py \
#    --train \
#    --base $config_yaml \
#    --gpus $gpu_list \
#    --logdir LOGS_DIR \
#    --name "$TRAIN_NAME"\
#    --scale_lr True \
#    --num_nodes 1 \
#    --check_val_every_n_epoch 10 \
#    --finetune_from $ckpt_path \
#    --every_n_train_steps 100 \
#data.params.batch_size=$BATCH_SIZE \
#lightning.trainer.accumulate_grad_batches=$ACCUMULATE_BATCHES \
#data.params.validation.params.n_gpus=$N_GPUS

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
# from: https://github.com/justinpinkney/stable-diffusion/blob/main/configs/stable-diffusion/pokemon.yaml

# train:       every_n_train_steps: 2000
# test:       every_n_train_steps: 100
cat <<EOT >$config_yaml
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 1
    num_workers: 4
    #num_val_workers: 0 # Avoid a weird val dataloader issue
    train:
      target: ldm.data.simple.hf_dataset
      params:
        name: /home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/txt2image_dataset_img_processed_tct_tras_sim_scored_filered_merged_1668610069_1669047469
        image_column: album_image
        text_column: filterd_tct_tags
        image_transforms:
        - target: torchvision.transforms.Resize
          params:
            size: 512
            interpolation: 3
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
  find_unused_parameters: False

  modelcheckpoint:
    params:
      every_n_epochs: 1 #every_n_train_steps: 8000
      save_top_k: -1
      monitor: null

  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 2000
        max_images: 4
        increase_log_steps: False
        log_first_step: True
        log_all_val: True
        log_images_kwargs:
          use_ema_scope: True
          inpaint: False
          plot_progressive_rows: False
          plot_diffusion_rows: False
          N: 4
          unconditional_guidance_scale: 3.0
          unconditional_guidance_label: [""]

  trainer:
    benchmark: True
    num_sanity_val_steps: 0
    accumulate_grad_batches: 4
model:
  base_learning_rate: 1.0e-04
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "image"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: crossattn
    scale_factor: 0.18215

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 1 ] # NOTE for resuming. use 10000 if starting from scratch
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      ckpt_path: "models/first_stage_models/kl-f8/model.ckpt"
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
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder

EOT

#BATCH_SIZE=4
N_GPUS=4
#ACCUMULATE_BATCHES=1
TRAIN_NAME=finetune_redbook_txt2img
gpu_list=0,1,2,3

# following: https://github.com/lyogavin/justinpinkney-stable-diffusion
#curl -L -x socks5h://localhost:8123 https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned/resolve/main/sd-clip-vit-l14-img-embed_ema_only.ckpt -o ../saved_models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt

ckpt_path=/home/ubuntu/cloudfs/saved_models/models--CompVis--stable-diffusion-v-1-4-original/snapshots/f0bb45b49990512c454cf2c5670b0952ef2f9c71/sd-v1-4-full-ema.ckpt

# testing setup
#    --every_n_train_steps 100 \
export CMD=" main.py \
    --train \
    --base $config_yaml \
    --gpus $gpu_list \
    --logdir LOGS_DIR \
    --name $TRAIN_NAME    \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from $ckpt_path \
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

/NEW_EDS/JJ_Group/zhuzr/consistency_models/cd_bedroom256_l2.pt
/NEW_EDS/JJ_Group/zhuzr/consistency_models/cd_bedroom256_lpips.pt

python image_sample.py --training_mode edm --generator determ-indiv --batch_size 8 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras


/home/xutd/.cache/torch/hub/checkpoints/


python image_sample.py --training_mode edm --generator determ-indiv --batch_size 1 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras

torch.Size([1, 1536, 1024])
torch.Size([1, 1024, 3, 8, 64])


torch.Size([2, 1536, 1024])
torch.Size([2, 1024, 3, 8, 64])
torch.Size([2, 1024, 8, 64])

import torch
from flash_attn.flash_attention import FlashAttention

out = inner_attn(torch.zeros([1,1024,3,8,64], dtype=torch.float16).cuda())

torch.Size([1, 1024, 8, 64])

torch.Size([2, 1024, 3, 8, 64])
torch.Size([2, 1024, 8, 64])


python image_sample.py --training_mode edm --generator determ-indiv --batch_size 1 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 1000 --sampler ancestral --model_path edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 1 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras


python image_inverse.py --training_mode edm --generator determ-indiv --batch_size 1 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 1000 --sampler sample_euler_ancestral_dps --model_path edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm False --weight_schedule karras --cfg super_resolution_config.yaml

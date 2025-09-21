# Extract Features from Diffusion Process
# Implementation of DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# Based on HyperFeature
# By Yuxiang Ji

import numpy as np
import os
from PIL import Image
import PIL
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler, 
    DDIMInverseScheduler,
    StableDiffusionPipeline,
    StableDiffusionSAGPipeline,
    StableDiffusionDepth2ImgPipeline
)
from transformers import (
    CLIPModel, 
    CLIPTextModel, 
    CLIPTokenizer
)
from archs.stable_diffusion.resnet import set_timestep, collect_feats_resnet, collect_feats_ca
"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""
'''
get_tokens_embedding: 텍스트 prompt를 CLIP 모델이 이해할 수 있는 벡터로 embedding
입력: 
  - clip_tokenizer: 텍스트를 토큰으로 변환
  - cip: 텍스트 모델
  - prompt
출력:
  - tokens
  - embedding 텐서 [B, 77(토큰), 768(임베딩 차원)]
'''
def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
  tokens = clip_tokenizer(
    prompt,
    padding="max_length",  #문장이 짧으면 최대 길이에 맞춰 padding
    max_length=clip_tokenizer.model_max_length,   #최대 길이
    truncation=True,    #최대 길이를 넘으면 자름
    return_tensors="pt",   #결과를 파이토치 텐서 형태로 바꿈
    return_overflowing_tokens=True,    #잘려나간 토큰 정보 반환
  )
  input_ids = tokens.input_ids.to(device)  #토큰 id를 gpu로
  embedding = clip(input_ids).last_hidden_state  #토큰 id를 CLIP 모델에 입력해 임베딩 벡터를 얻음
  return tokens, embedding

'''
latent_to_image: latent의 텐선를 image로 변환
입력:
  - VAE
  - latent 텐서 [B, C, H, W] 
출력:
  - image
'''

def latent_to_image(vae, latent):
  latent = latent / 0.18215   #스케일 이전의 값으로 되돌려 놓음
  image = vae.decode(latent.to(vae.dtype)).sample  #VAE 디코더를 통해 latent 텐서를 픽셀 공간에 복원
  image = (image / 2 + 0.5).clamp(0, 1)  #이미지 픽셀값의 범위를 [-1, 1]에서 [0, 1]로 정규화
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()  #텐서를 cpu로 옮기고, 계산 그래프에서 분리(detach)한 뒤, [B, C, H, W] -> [B, H, W, C]
  image = (image[0] * 255).round().astype("uint8")  #픽셀 범위[0, 1] -> [0, 255] 정수형으로 바꿈
  image = Image.fromarray(image)  #넘파이 배열을 PIL 이미지 객체로 변환
  return image

''''
image_to_latent: image -> latent 텐서로 변환
입력:
  - VAE
  - Image
  - generator: 재현성을 위한 랜덤 시드 생성기
  - w, h: 변환할 이미지 크기
출력:
  - latent 텐서 [B, 4, 64, 64]
'''
def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)   #이미지 크기를 지정된 크기 (w, h)로 조절
  image = np.array(image).astype(np.float32)  #넘파이 배열로 변환
  # remove alpha channel: 이미지에 투명도(alpha channel)이 있을 경우, 제거하고 RGB 채널만 남김
  if len(image.shape) == 2:   #흑백 이미지일 경우
    image = image[:, :, None]
  else:   #컬러이미지일 경우
    image = image[:, :, (0, 1, 2)]  
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)  #[H, W, C]에서 [B, C, W, H] 형태로 바꿈 (B = 1)
  image = torch.from_numpy(image)  #넘파이 배열을 텐서로 변경
  image = image / 255.0    #픽셀의 범위를 [0, 255] -> [0.0, 1.0]
  image = 2. * image - 1.     #픽셀 범위 [0.0, 1.0] -> [-1, 1]
  image = image.to(vae.device)
  image = image.to(vae.dtype)
  return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215  #VAE 인코더를 통해 latent 텐서로 변환 후 스케일링 값 곱함

'''
get_xt_next: DDIM 샘플링 공식을 사용해, 현재 timestep의 노이즈가 낀 latent(x_t)에서 다음 타임스텝의 x_t+1을 계산 
입력: 
  - xt: 현재 타임스텝 t에서의 latent
  - et: U-Net이 예측한 timestep t에서의 noise
  - at, at_next: 타임스텝 t와 t-1에서의 노이즈 스케줄(alpha_cumprod)
  - ~~ : 세부 파라미터

출력: 
  - x0_t: 예측된 노이즈가 제거된 원본 이지미의 latent 텐서
  - xt_next: 다음 타임스텝 t-1에서의 latent 텐서
'''

def get_xt_next(xt, et, at, at_next, a_skip, eta, tmask, do_adpm_steps=False, gamma_t=None):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  #DDIM 공식: 예측된 노이즈(et)를 이용해 노이즈가 없는 x0_t를 추정
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - a_skip) * (1 - at_next) / (1 - at)).sqrt()
    )
    # c1 = eta * (1 - a_skip).sqrt()
  c2 = torch.max((1 - at_next) - (c1 * tmask) ** 2, torch.Tensor([1e-10]).to(et.device))[0].sqrt()
  # print(f'c1={c1}, c2={c2}, at={at}, at_next={at_next}, c12={c1**2}')
  if do_adpm_steps and eta > 0.0:
    cov_x_0_pred = (1 - at) / at * (1. - gamma_t)
    cov_x_0_pred_clamp = torch.clamp(cov_x_0_pred, 0., 1.)
    coeff_cov_x_0 = (at_next ** 0.5 - ((c2 ** 2) * at / (1 - at)) ** 0.5) ** 2
    offset = coeff_cov_x_0 * cov_x_0_pred_clamp
    c1 = (c1 ** 2 + offset) ** 0.5
    # print(f'cov_x_0_pred={cov_x_0_pred}, coef={coeff_cov_x_0}, c1={c1}, c2={c2}, at={at}, at_next={at_next}, c12={c1**2}')
  xt_next = at_next.sqrt() * x0_t + c1 * tmask * torch.randn_like(et) + c2 * et
  # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn(1).to(device=et.device, dtype=et.dtype) * tmask * torch.randn_like(et) + c2 * et
  return x0_t, xt_next

'''
generalized_steps: diffusion 과정을 수행하고, 최종 latent를 반환
입력:
  -x: 시작 latent 텐서
  -model: U-Net 모델
  -scheduler: DDIM 스케줄러
  - **kwargs: 다양한 옵션을 dict 형태로 받음 ex) prompt, guidance_scale 등
출력:
  - xs: diffusion 과정이 끝난 최종 latent 텐서
'''
def generalized_steps(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  training = kwargs.get("training", False)

  seq = scheduler.timesteps   #스케줄러에서 Timestep 시퀀스를 가져옴 ex)[T,...,0]
  seq = torch.flip(seq, dims=(0,))  #timestep 뒤집음
  b = scheduler.betas   #noise 비율
  b = b.to(x.device)

  batch_size = x.shape[0]   #x[B, C, H, W] 
  x_h = x.shape[2]
  x_w = x.shape[3]
  x_c = x.shape[1]

  images = kwargs.get("images", None)   #kwargs에서 이미지(RGB) 가져옴 (depth-guided 등에 사용)
  
  
  with torch.no_grad():  #그라디언트 계산x
    n = x.size(0)   #배치 차원 크기
    seq_next = [0] + list(seq[:-1])  #다음 타임스텝 시퀀스를 만듦 ex) seq = [T, T-1,...] -> seq_next = [0, T, T-1,...]

    #이미지 -> 노이즈(inversion) or 노이즈 -> 이미지 (generation)
    if kwargs.get("run_inversion", False):   #Inversion
      seq_iter = seq_next
      seq_next_iter = seq
      do_inversion = True
    else:
      seq_iter = reversed(seq)
      seq_next_iter = reversed(seq_next)
      do_inversion = False

    do_one_step = kwargs.get("do_one_step", False)
    if do_one_step and kwargs.get("run_inversion", False):
      seq_iter = list(seq[:])
      seq_next_iter = seq_iter
      t = torch.tensor(seq[0], dtype=torch.long, device=x.device)
      noise = torch.randn_like(x).to(x.device)
      x = scheduler.add_noise(x, noise, t)

    do_optim_steps = kwargs.get("do_optim_steps", False)
    beta1 = 0.7
    beta2 = 0.77
    mt = 0
    vt = 0
    s_tmin = kwargs.get("s_tmin")
    s_tmax = kwargs.get("s_tmax")


    tmask = torch.ones((x.shape[2], x.shape[3])).to(device=x.device, dtype=x.dtype)

    #[option] depth 정보를 사용하는 경우
    do_with_depth = kwargs.get("do_with_depth", False)
    feature_extractor = kwargs.get("feature_extractor", None)
    depth_estimator = kwargs.get("depth_estimator", None)
    if do_with_depth:
      images = F.interpolate(images, size=(384, 384), mode='bilinear')
      depth_map = depth_estimator(images).predicted_depth
      depth_map = depth_map[:, None, ...]
      depth_map = F.interpolate(depth_map, size=(x_h, x_w), mode='bilinear')
      depth_min, depth_max = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True), torch.amax(depth_map, dim=[1, 2, 3],
                                                                                           keepdim=True)
      depth_map = 2. * (depth_map - depth_min) / (depth_max - depth_min) - 1.
    
    #[option] 마스크(segmentation Map)을 사용하는 경우 (논문)
    do_mask_steps = kwargs.get("do_mask_steps", False)
    if do_mask_steps:
      ref_semantic_seg = kwargs.get("gt_semantic_seg", None)   #외부에서 입력된 정답 segmentation map
      label_map = kwargs.get("label_map")   #라벨 번호와 텍스트를 매핑한 dict / ex) {1: "a photo of a car"}
      clip_tokenizer = kwargs.get("clip_tokenizer")
      clip = kwargs.get("clip")
      
      #특정 타임스텝 범위에서만 마스크를 적용하기 위한 설정
      mask_min = kwargs.get("mask_min", 0)
      mask_max = kwargs.get("mask_max", 1000)

      ###### Mask [len(prompts) * batch_size, 1, h, w]
      #기본 마스크(전체가 1인) 생성 [1, B, 1, H, W]
      masks = torch.ones((1, batch_size, 1, x.shape[2], x.shape[3])).to(dtype=x.dtype, device=x.device)
      prompts = [kwargs["prompt"]] * batch_size   #기본 프롬프트를 배치 크기만큼 복사
      prompt_cnt = 1
      
      if ref_semantic_seg != None:   #정답 segmentation map이 주어진 경우, 각 클래스별로 mask와 prompt 생성
        mask_unique = torch.unique(ref_semantic_seg)  #map에 존재하는 모든 class의 label을 찾음
        for label in mask_unique:  #각 라벨에 대해 반복
          if label.item() not in label_map.keys():  
            continue   #모르는 라벨이면 걍 건너 뜀
          mask = (ref_semantic_seg == label).to(dtype=x.dtype)   #해당 라벨에 해당하는 영역만 1인 마스크를 생성 (논문에서 M에 해당)
          if len(mask.shape) == 3:
            mask = mask[:, None, :, :]
          mask = F.interpolate(mask, size=(x_h, x_w), mode='nearest')
          mask = mask[None, ...]
          
          masks = torch.cat([masks, mask], dim=0)  #최종 마스크 목록에 추가

          label_text = label_map[label.item()]   #라벨에 해당하는 텍스트 프롬프트를 가져옴
          prompts += [label_text] * batch_size  #프롬프트 목록에 추가
          prompt_cnt += 1
        
        #생성된 모든 프롬프트를 한 번에 임베딩으로 변환
        _, label_embedding = get_tokens_embedding(clip_tokenizer, clip, x.device, prompts)
        uncond_embedding = kwargs["unconditional"]  #비조건부 임베딩 (보통 빈 텍스트)
        #최종 텍스트 임베딩: [비조건부, 조건부1, 조건부2, ...] 순서로 합침
        text_embedding = torch.cat([uncond_embedding, label_embedding], dim=0)
        # print('shape', text_embedding.shape)

        count = torch.zeros_like(x)
        value = torch.zeros_like(x)
    #---------diffusion 과정------------
    xs = x   #현재 latent xs 생성
    for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
      max_i = kwargs.get("max_i", None)
      min_i = kwargs.get("min_i", None)
      if max_i is not None and i >= max_i:
        break
      if min_i is not None and i < min_i:
        continue
      
      #현재/다음 timestep에 해당하는 노이즈 스케줄(alpha_cumprod) 값을 가져옴
      t = (torch.ones(1) * t).to(x.device)
      next_t = (torch.ones(1) * next_t).to(x.device)

      at = (1 - b).cumprod(dim=0).index_select(0, t.long())
      at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
      
      # Expand to the correct dim
      at, at_next = at[:, None, None, None], at_next[:, None, None, None]

      set_timestep(model, i)   #현재 timestep 번호를 U-net에 알려줌
      
      xt = xs  #현재 latent
      if do_with_depth:
        xt_input = torch.cat((xt, depth_map), dim=1)
      else:
        xt_input = xt
      # xt = torch.cat((xs, depth_map), dim=1)
      cond = kwargs["conditional"]  #조건부 텍스트 인베딩
      guidance_scale = kwargs.get("guidance_scale", -1)   #Classifier-Free Guidance 강도

      #-------U-Net으로 노이즈 예측----------
      if do_mask_steps and t >= mask_min and t <= mask_max:  #마스크를 사용하는 경우, 여러 프롬프트에 대해 한번에 노이즈 예측
        if ref_semantic_seg != None:
          xt_input = xt_input.repeat(prompt_cnt + 1, 1, 1, 1)  #입력 latent를 prompt 개수만큼 복사함(P개의 프롬프트와 1개의 uncond prompt) [B, C, H, W] -> [(P+1)*B, C, H, W]
          #모델에 복사된 입력과 전체 text embedding을 넣어 노이즈(et)를 예측
          et = model(xt_input, t, encoder_hidden_states=text_embedding).sample
          # 예측된 노이즈를 분리하고, Classifier-Free Guidance 적용 (Text prompt가 없을 때)
          et_uncond = et[:batch_size, ...].repeat(prompt_cnt, 1, 1, 1)
          et_text = et[batch_size:, ...]
          et = et_uncond + guidance_scale * (et_text - et_uncond)
        else:  #일반적인 classifier-Free guidance: 입력을 2번 반복하고, uncond / cond 예측을 조합해 최종 et 계산
          uncond = kwargs["unconditional"]
          xt_input = xt_input.repeat(2, 1, 1, 1)
          encoder_hidden_states = torch.cat([uncond, cond], dim=0)
          et = model(xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
          et_uncond = et[:batch_size, ...]
          et_cond = et[batch_size:, ...]
          et = et_uncond + guidance_scale * (et_cond - et_uncond)
      elif guidance_scale == -1:
        et = model(xt, t, encoder_hidden_states=cond).sample
      else:
        # If using Classifier-Free Guidance, the saved feature maps
        # will be from the last call to the model, the conditional prediction
        uncond = kwargs["unconditional"]
        xt_input = xt_input.repeat(2, 1, 1, 1)
        encoder_hidden_states = torch.cat([uncond, cond], dim=0)
        et = model(xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
        et_uncond = et[:batch_size, ...]
        et_cond = et[batch_size:, ...]
        et = et_uncond + guidance_scale * (et_cond - et_uncond)
      
      #------다음 step 계산
      eta = kwargs.get("eta", 0.0)
      if t > s_tmin and t < s_tmax:
        eta = eta
      else:
        eta = 0.0

      if t > next_t:
        a_skip = at / at_next
      else:
        a_skip = at_next / at

      if do_mask_steps and ref_semantic_seg != None:  #마스크를 사용하는 경우, 각 마스크 영역별로 생성된 결과를 가중 평균
        if t >= mask_min and t <= mask_max:           #논문에서 I_t+1 구하는 그 부분인듯
          
          xts = xt.repeat(prompt_cnt, 1, 1, 1)
          x0_ts = (xts - et * (1 - at).sqrt()) / at.sqrt()

          c1 = (
            eta * ((1 - a_skip) * (1 - at_next) / (1 - at)).sqrt()
          )
          c2 = torch.max((1 - at_next) - (c1 * tmask) ** 2, torch.Tensor([1e-10]).to(et.device))[0].sqrt()
          #xts_next: P개의 프롬프트 각각에 대한 예측 latent 모임
          xts_next = at_next.sqrt() * x0_ts + c1 * tmask * torch.randn_like(et) + c2 * et

        # masks의 shape를 xts_next와 곱할 수 있도록 변경한 후, 곱함
        ###### [prompt_cnt*batch_size, c, h, w] * [prompt_cnt*batch_size, 1, h, w] -> [prompt_cnt, batch_size, c, h, w] -> [batch_size, c, h, w]
        value = (xts_next * masks.view(prompt_cnt*batch_size, 1, xts.shape[2], xts.shape[3])).view(prompt_cnt, batch_size, x_c, x_h, x_w).sum(dim=0)
        ###### [prompt_cnt, batch_size, 1, h, w] -> [batch_size, 1, h, w]
        count = masks.sum(dim=0)
        #마스크가 적용된 영역은 value를 count로 나누어 평균을 냄
        xt_next = torch.where(count > 0, value / count, value)
      else:
        _, xt_next = get_xt_next(xt, et, at, at_next, a_skip, eta, tmask)

      xs = xt_next

    return xs  #최종 latent 반환
  
  '''
  n = x.size(0)   #배치 차원 크기
  seq_next = [0] + list(seq[:-1])  #다음 타임스텝 시퀀스를 만듦 ex) seq = [T, T-1,...] -> seq_next = [0, T, T-1,...]

  #이미지 -> 노이즈(inversion) or 노이즈 -> 이미지 (generation)
  if kwargs.get("run_inversion", False):   #Inversion
    seq_iter = seq_next
    seq_next_iter = seq
    do_inversion = True
  else:
    seq_iter = reversed(seq)
    seq_next_iter = reversed(seq_next)
    do_inversion = False

  do_one_step = kwargs.get("do_one_step", False)
  if do_one_step and kwargs.get("run_inversion", False):
    seq_iter = list(seq[:])
    seq_next_iter = seq_iter
    t = torch.tensor(seq[0], dtype=torch.long, device=x.device)
    noise = torch.randn_like(x).to(x.device)
    x = scheduler.add_noise(x, noise, t)

  do_optim_steps = kwargs.get("do_optim_steps", False)
  beta1 = 0.7
  beta2 = 0.77
  mt = 0
  vt = 0
  s_tmin = kwargs.get("s_tmin")
  s_tmax = kwargs.get("s_tmax")


  tmask = torch.ones((x.shape[2], x.shape[3])).to(device=x.device, dtype=x.dtype)

  #[option] depth 정보를 사용하는 경우
  do_with_depth = kwargs.get("do_with_depth", False)
  feature_extractor = kwargs.get("feature_extractor", None)
  depth_estimator = kwargs.get("depth_estimator", None)
  if do_with_depth:
    images = F.interpolate(images, size=(384, 384), mode='bilinear')
    depth_map = depth_estimator(images).predicted_depth
    depth_map = depth_map[:, None, ...]
    depth_map = F.interpolate(depth_map, size=(x_h, x_w), mode='bilinear')
    depth_min, depth_max = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True), torch.amax(depth_map, dim=[1, 2, 3],
                                                                                           keepdim=True)
    depth_map = 2. * (depth_map - depth_min) / (depth_max - depth_min) - 1.
    
  #[option] 마스크(segmentation Map)을 사용하는 경우 (논문)
  do_mask_steps = kwargs.get("do_mask_steps", False)
  if do_mask_steps:
    ref_semantic_seg = kwargs.get("gt_semantic_seg", None)   #외부에서 입력된 정답 segmentation map
    label_map = kwargs.get("label_map")   #라벨 번호와 텍스트를 매핑한 dict / ex) {1: "a photo of a car"}
    clip_tokenizer = kwargs.get("clip_tokenizer")
    clip = kwargs.get("clip")
      
    #특정 타임스텝 범위에서만 마스크를 적용하기 위한 설정
    mask_min = kwargs.get("mask_min", 0)
    mask_max = kwargs.get("mask_max", 1000)

    ###### Mask [len(prompts) * batch_size, 1, h, w]
    #기본 마스크(전체가 1인) 생성 [1, B, 1, H, W]
    masks = torch.ones((1, batch_size, 1, x.shape[2], x.shape[3])).to(dtype=x.dtype, device=x.device)
    prompts = [kwargs["prompt"]] * batch_size   #기본 프롬프트를 배치 크기만큼 복사
    prompt_cnt = 1
      
    if ref_semantic_seg != None:   #정답 segmentation map이 주어진 경우, 각 클래스별로 mask와 prompt 생성
      mask_unique = torch.unique(ref_semantic_seg)  #map에 존재하는 모든 class의 label을 찾음
      for label in mask_unique:  #각 라벨에 대해 반복
        if label.item() not in label_map.keys():  
          continue   #모르는 라벨이면 걍 건너 뜀
        mask = (ref_semantic_seg == label).to(dtype=x.dtype)   #해당 라벨에 해당하는 영역만 1인 마스크를 생성 (논문에서 M에 해당)
        if len(mask.shape) == 3:
          mask = mask[:, None, :, :]
        mask = F.interpolate(mask, size=(x_h, x_w), mode='nearest')
        mask = mask[None, ...]
          
        masks = torch.cat([masks, mask], dim=0)  #최종 마스크 목록에 추가

        label_text = label_map[label.item()]   #라벨에 해당하는 텍스트 프롬프트를 가져옴
        prompts += [label_text] * batch_size  #프롬프트 목록에 추가
        prompt_cnt += 1
        
      #생성된 모든 프롬프트를 한 번에 임베딩으로 변환
      _, label_embedding = get_tokens_embedding(clip_tokenizer, clip, x.device, prompts)
      uncond_embedding = kwargs["unconditional"]  #비조건부 임베딩 (보통 빈 텍스트)
      #최종 텍스트 임베딩: [비조건부, 조건부1, 조건부2, ...] 순서로 합침
      text_embedding = torch.cat([uncond_embedding, label_embedding], dim=0)
      # print('shape', text_embedding.shape)

      count = torch.zeros_like(x)
      value = torch.zeros_like(x)
      if ref_semantic_seg is not None:
        weighted_masks = WeightMask(prompt_cnt, batch_size, x_h, x_w).to(x.device)
  #---------diffusion 과정------------
  xs = x   #현재 latent xs 생성
  for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
    max_i = kwargs.get("max_i", None)
    min_i = kwargs.get("min_i", None)
    if max_i is not None and i >= max_i:
      break
    if min_i is not None and i < min_i:
      continue
      
    #현재/다음 timestep에 해당하는 노이즈 스케줄(alpha_cumprod) 값을 가져옴
    t = (torch.ones(1) * t).to(x.device)
    next_t = (torch.ones(1) * next_t).to(x.device)

    at = (1 - b).cumprod(dim=0).index_select(0, t.long())
    at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
      
    # Expand to the correct dim
    at, at_next = at[:, None, None, None], at_next[:, None, None, None]

    set_timestep(model, i)   #현재 timestep 번호를 U-net에 알려줌
      
    xt = xs  #현재 latent
    if do_with_depth:
      xt_input = torch.cat((xt, depth_map), dim=1)
    else:
      xt_input = xt
    # xt = torch.cat((xs, depth_map), dim=1)
    cond = kwargs["conditional"]  #조건부 텍스트 인베딩
    guidance_scale = kwargs.get("guidance_scale", -1)   #Classifier-Free Guidance 강도

    #-------U-Net으로 노이즈 예측----------
    if do_mask_steps and t >= mask_min and t <= mask_max:  #마스크를 사용하는 경우, 여러 프롬프트에 대해 한번에 노이즈 예측
      if ref_semantic_seg != None:
        xt_input = xt_input.repeat(prompt_cnt + 1, 1, 1, 1)  #입력 latent를 prompt 개수만큼 복사함(P개의 프롬프트와 1개의 uncond prompt) [B, C, H, W] -> [(P+1)*B, C, H, W]
        #모델에 복사된 입력과 전체 text embedding을 넣어 노이즈(et)를 예측
        et = model(xt_input, t, encoder_hidden_states=text_embedding).sample
        # 예측된 노이즈를 분리하고, Classifier-Free Guidance 적용 (Text prompt가 없을 때)
        et_uncond = et[:batch_size, ...].repeat(prompt_cnt, 1, 1, 1)
        et_text = et[batch_size:, ...]
        et = et_uncond + guidance_scale * (et_text - et_uncond)
      else:  #일반적인 classifier-Free guidance: 입력을 2번 반복하고, uncond / cond 예측을 조합해 최종 et 계산
        uncond = kwargs["unconditional"]
        xt_input = xt_input.repeat(2, 1, 1, 1)
        encoder_hidden_states = torch.cat([uncond, cond], dim=0)
        et = model(xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
        et_uncond = et[:batch_size, ...]
        et_cond = et[batch_size:, ...]
        et = et_uncond + guidance_scale * (et_cond - et_uncond)
    elif guidance_scale == -1:
      et = model(xt, t, encoder_hidden_states=cond).sample
    else:
      # If using Classifier-Free Guidance, the saved feature maps
      # will be from the last call to the model, the conditional prediction
      uncond = kwargs["unconditional"]
      xt_input = xt_input.repeat(2, 1, 1, 1)
      encoder_hidden_states = torch.cat([uncond, cond], dim=0)
      et = model(xt_input, t, encoder_hidden_states=encoder_hidden_states).sample
      et_uncond = et[:batch_size, ...]
      et_cond = et[batch_size:, ...]
      et = et_uncond + guidance_scale * (et_cond - et_uncond)
      
    #------다음 step 계산
    eta = kwargs.get("eta", 0.0)
    if t > s_tmin and t < s_tmax:
      eta = eta
    else:
      eta = 0.0

    if t > next_t:
      a_skip = at / at_next
    else:
      a_skip = at_next / at

    if do_mask_steps and ref_semantic_seg != None:  #마스크를 사용하는 경우, 각 마스크 영역별로 생성된 결과를 가중 평균
      if t >= mask_min and t <= mask_max:           #논문에서 I_t+1 구하는 그 부분인듯
          
        xts = xt.repeat(prompt_cnt, 1, 1, 1)
        x0_ts = (xts - et * (1 - at).sqrt()) / at.sqrt()

        c1 = (
          eta * ((1 - a_skip) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = torch.max((1 - at_next) - (c1 * tmask) ** 2, torch.Tensor([1e-10]).to(et.device))[0].sqrt()
        #xts_next: P개의 프롬프트 각각에 대한 예측 latent 모임
        xts_next = at_next.sqrt() * x0_ts + c1 * tmask * torch.randn_like(et) + c2 * et

      value = xts_next * masks.view(prompt_cnt * batch_size, 1, xts.shape[2], xts.shape[3])
      weighted_value = weighted_masks.forward(value)
      count = masks.sum(dim = 0)
      xt_next = torch.where(count > 0, weighted_value / count, weighted_value)
      
      
      # masks의 shape를 xts_next와 곱할 수 있도록 변경한 후, 곱함
      ###### [prompt_cnt*batch_size, c, h, w] * [prompt_cnt*batch_size, 1, h, w] -> [prompt_cnt, batch_size, c, h, w] -> [batch_size, c, h, w]
      #value = (xts_next * masks.view(prompt_cnt*batch_size, 1, xts.shape[2], xts.shape[3])).view(prompt_cnt, batch_size, x_c, x_h, x_w).sum(dim=0)
      ###### [prompt_cnt, batch_size, 1, h, w] -> [batch_size, 1, h, w]
      #count = masks.sum(dim=0)
      #마스크가 적용된 영역은 value를 count로 나누어 평균을 냄
      #xt_next = torch.where(count > 0, value / count, value)
      
    else:
      _, xt_next = get_xt_next(xt, et, at, at_next, a_skip, eta, tmask)

    xs = xt_next

  return xs  #최종 latent 반환
  '''
  
'''
ref_segmantic_seg_to_masks: GT 세그멘테이션 map(ref_segmantic_seg)을 받아, 각 클래스별 이진 마스크로 변환
입력:
  - ref_semantic_seg: GT segmentation map
  - label_map: 클래스 번호와 텍스트 설명을 연결해주는 dict 
  - x: 현재 처리 중인 latent 텐서
출력:
  - masks:여러 개의 이진 마스크가 겹쳐진 텐서 [클래스 개수 + 1, 1, latent 높이, latent 넓이]
'''
def ref_semantic_seg_to_masks(ref_semantic_seg, label_map, x):
  masks = torch.ones((1, 1, x.shape[2], x.shape[3])).to(dtype=x.dtype, device=x.device)  #전체 영역이 1로 채워진 기본 마스크를 만듦(배경)
  if ref_semantic_seg != None:
    mask_unique = torch.unique(ref_semantic_seg)  #세그멘테이션 map에 존재하는 모든 고유한 클래스 번호를 찾아냄
    for label in mask_unique:  #각 라벨에 대해 반복작업
      if label.item() not in label_map.keys():
        continue
      mask = (ref_semantic_seg == label).to(dtype=x.dtype)  #현재 클래스에 해당하는 픽셀은 1, 나머지는 0처리
      mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='nearest')  #x와 같은 크기로 리사이징
      masks = torch.cat([masks, mask], dim=0) #(1, 1, H, W) -> (2, 1, H, W) -> (3, 1, H, W) -> ...
  return masks


def freeze_weights(weights):
  for param in weights.parameters():
    param.requires_grad = False

''''
init_models: stable diffusion 초기화
입력:
  - device
  - model_id
  - freeze: True
출력:
  - pipe, unet, vae, clip, clip_tokenizer
'''
def init_models(
    unet,
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze_clip = True,
    freeze_vae = True,
    freeze_unet = True,
    do_with_depth=False,
  ):
  if 'depth' in model_id:
    do_with_depth = True
  if do_with_depth:
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
      model_id,
      revision="fp16",
      torch_dtype=torch.float16,
    )
  else:
    pipe = StableDiffusionPipeline.from_pretrained(
      model_id,
      revision="fp16",
      torch_dtype=torch.float16,
    )
  pipe.unet = unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  unet.to(device)
  vae.to(device)
  clip.to(device)
  
  if freeze_unet:
    freeze_weights(unet)
  if freeze_vae:
    freeze_weights(vae)
  if freeze_clip:
    freeze_weights(clip)
  return pipe, unet, vae, clip, clip_tokenizer


def collect_and_resize_feats(unet, idxs, timestep, resolution=None):
  latent_feats = collect_feats(unet, idxs=idxs)
  latent_feats = [feat[timestep] for feat in latent_feats]
  if resolution != None:
      latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear") for latent_feat in latent_feats]
  # latent_feats = torch.cat(latent_feats, dim=1)
  return latent_feats

'''
get_stride_num: feature를 뽑을 레이어 인덱스 목록(idxs)를 받아서, U-Net의 각 해상도 레벨(stride)별로
                몇 개의 feature를 추출하는지 개수를 셈
입력:
   - idxs: feature를 뽑을 레이어의 인덱스 목록  ex) [[1, 0], [1, 1], [2,0]] -> 1번에서 2개, 2번에서 1개
출력:
   - cnt[0], cnt[1], cnt[2]: 1, 2, 3번 레벨에서 추출한 피처의 개수를 반환
'''
def get_stride_num(idxs):
  cnt = [0 for _ in range(3)]
  for [i, j] in idxs:
    cnt[i - 1] += 1
  return cnt[0], cnt[1], cnt[2]

'''
Collect_stride_feats_with_timesteplist: 지정된 레이어(idx)와 타임스텝(timestep_list)에서 feature 수집
                                        + 이를 concat하여 최종 feature map을 만듦 
입력:
   - Unet
   - idxs_resnet: feature를 뽑을 ResNet 블록의 인덱스 목록
   - idxs_ca: feature를 추출할 Cross-Attention 블록의 인덱스 목록
   - timestep_list: feature를 추출할 타임스텝 목록
출력:
   - 4개의 텐서: U-net의 각 해상도 레벨(stride)에서 추출되고 결합된 최종 feature map들
'''
def collect_stride_feats_with_timesteplist(unet, idxs_resnet, idxs_ca, timestep_list, do_mask_steps=False,
                                           x=None, ref_semantic_seg=None, label_map=None, guidance_scale=0.0):
  batch_size = x.shape[0]  
  #U-net의 각 stride별 featuer map의 H와 W를 저장할 리스트 초기화 
  latent_h = [0, 0, 0, 0]  
  latent_w = [0, 0, 0, 0]
  
  #U-Net의 ResNet 블록에서 feature 수집
  latent_feats_resnet = collect_feats_resnet(unet, idxs=idxs_resnet)   
  latent_feats_resnet_idxs = []  #수집한 feature들을 레이어/타임스텝별로 정리할 빈 리스트를 만듦
  #인덱스 목록(idxs_resnet)과 실제 feature 데이터(latent_feats_resnet)를 짝지어 반복
  for idx, feat in zip(idxs_resnet, latent_feats_resnet):
    latents_feats_idxs_t = []  #특정 레이어에서 나온 여러 타임스텝의 feature을 담을 임시 리스트
    for timestep in timestep_list:  #timestep_list에서 feature 추출
      feat_t = feat[timestep]  #현재 레이어(feat)의 결과물에서 현재 타임스텝에 해당하는 feature 텐서를 가져옴
      feat_t = feat_t[:batch_size, ...]  #classifier-free guidance를 사용하면 배치가 2배(uncond, cond)로 들어가므로, 실제 bach_size만큼만 잘라내 conditional feature만 사용
      latents_feats_idxs_t.append(feat_t)  #해당 타임스텝의 feature를 임시 리스트에 추가
    latent_feats_resnet_idxs.append(latents_feats_idxs_t)  #한 레이어에서 모든 타임스텝의 feature를 다 모았다면, 해당 리스트를 최종 리스트에 추가
    #현재 레이어의 크기 저장 (idx[0]은 stride 레벨)
    latent_h[idx[0]] = feat_t.shape[2]  
    latent_w[idx[0]] = feat_t.shape[3]
  #채널  방향으로 텐서들을 이어 붙임 ex) (B, 320, H, W) 3개를 붙이면 (B, 960, H, W)  
  feats_cat_resnet_idxs = [torch.cat(feat_t, dim=1) for feat_t in latent_feats_resnet_idxs]
  # print('resnet shape', feats_cat_resnet_idxs[0].shape, feats_cat_resnet_idxs[3].shape, feats_cat_resnet_idxs[6].shape, feats_cat_resnet_idxs[9].shape)

  #CA 블록 피처 추출
  latent_feats_ca = collect_feats_ca(unet, idxs=idxs_ca)
  latent_feats_ca_idxs = []
  for idx, feat in zip(idxs_ca, latent_feats_ca):
    latents_feats_idxs_t = []
    for timestep in timestep_list:
      feat_t = feat[timestep]  #현재 타임스텝의 feature를 가져옴
      
      #CA feature는 (B*prompt_cnt, H*W, C) 형태로 저장되어 있음
      prompt_cnt = feat_t.shape[0] // batch_size  #프롬프트 개수(prompt_cnt) 계산
      feat_c = feat_t.shape[2]   #어텐션 feature의 채널(C)
      # print('??', feat_t.shape, prompt_cnt, batch_size, idx, latent_h[idx[0]], latent_w[idx[0]], feat_c)
      feat_t = feat_t.view(prompt_cnt, batch_size, latent_h[idx[0]], latent_w[idx[0]], feat_c)  #[B*prompt_cnt, H*W, C] -> [prompt_cnt, B, H, W, C]
      feat_t = feat_t.permute(1, 0, 4, 2, 3)  #[prompt_cnt, B, H, W, C] -> [B, prompt_cnt, C, H, W]
      feat_t = feat_t.sum(dim=1)  #prompt_cnt 차원에 따라 모든 값을 더함
      latents_feats_idxs_t.append(feat_t)
    latent_feats_ca_idxs.append(latents_feats_idxs_t)
  feats_cat_ca_idxs = [torch.cat(feat_t, dim=1) for feat_t in latent_feats_ca_idxs]  #타임스텝별로 수집된 특징들을 채널 방향으로 concat

  # print('ca shape', feats_cat_ca_idxs[0].shape, feats_cat_ca_idxs[3].shape, feats_cat_ca_idxs[6].shape)

  ### with CrossAttn & ResNet 특징 모두 사용
  #U-Net의 각 해상도 레벨(stride)별로 ResNet 피처와 CA 피처를 concat
  #총 4개의 level에서의 fusion(채널 방향으로 concat) feature map 4개 반환
  if len(idxs_ca) != 0 and len(idxs_resnet) != 0:
    return torch.cat(feats_cat_resnet_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[3:6]+feats_cat_ca_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[6:9]+feats_cat_ca_idxs[3:6], dim=1), torch.cat(feats_cat_resnet_idxs[9:12]+feats_cat_ca_idxs[6:9], dim=1) 

  elif len(idxs_ca) != 0 and len(idxs_resnet) == 0:
    return None, torch.cat(feats_cat_ca_idxs[0:3], dim=1), torch.cat(feats_cat_ca_idxs[3:6], dim=1), torch.cat(feats_cat_ca_idxs[6:9], dim=1)
  
  ### Only support idxs (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) now. Added by jyx
  elif len(idxs_resnet) != 0 and len(idxs_ca) == 0:
    return torch.cat(feats_cat_resnet_idxs[0:3], dim=1), torch.cat(feats_cat_resnet_idxs[3:6], dim=1), torch.cat(feats_cat_resnet_idxs[6:9], dim=1), torch.cat(feats_cat_resnet_idxs[9:12], dim=1) 


if __name__ == '__main__':
  pass
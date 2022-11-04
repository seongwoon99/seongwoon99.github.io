---
title: "SwAV을 이용한 ResNet Fine-tuning"
categories:
  - Project
tags:
  - self_supervised
  - swav
  - fine-tuning
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JEKYLL"
toc_icon: "bookmark"
---

### SwAV 이란
* SwAV은 Contrastive Learing 기반의 Self Supervised Learning 방법 중 하나로 label이 없는 데이터에 대하여 훈련을 진행할 수 있다.
* Contrastive Learning은 입력 샘플 간의 비교를 통해 학습을 수행하는 방법으로, SwAV을 통해서는 unlabeled 이미지에 대한 feature를 학습할 수 있다.

### SwAV Code

#### train.py 사전 처리
* SwAV으로 생성된 pth파일의 경우 multi gpu를 사용하기 때문에 레이어 이름에 `module`이란 명칭이 포함된다. 
* resnet으로 fine-tuning하기 위해 편의상 레이어 이름에 `module` 이 포함되지 않는 pth 파일을 저장하기 위해 다음과 같이 수정한다.  
* `main_swav.py` 내부 속 `259 line` 부근에 `state_dict`의 내용을 수정한다.  

```python
# save checkpoints  수정!!
if args.rank == 0:
    save_dict = {
        "epoch": epoch + 1,
        "state_dict": model.module.state_dict(), # module # 다 가져와서 거기서 매칭!
        "optimizer": optimizer.state_dict(),
    }
```

#### Training
##### train.py의 주요 인자
* `--data_pth` : 이미지 경로를 입력한다. 이때 해당 경로의 `하위 폴더를 추가로 만들어 저장한다.`
* `--epochs` : 400 또는 800을 추천하는 듯하다.
* `--dump_path` : output 저장할 경로, pth, queue, stats 등의 파일이 저장된다. 

```
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path ~/image_folder \
--epochs 800 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15 \
--dump_path ~/output_dir \ 
```

##### ImageNet 가중치 탑재 후 전이학습
* SwAV에서는 ImageNet으로 학습된 가중치를 제공한다. 
* ImageNet 데이터로 학습된 가중치 이후로 전이학습을 진행하기 위해서는 다음과 같은 코드를 추가한다. 
* `main_swav.py` 내부의 `nn.parallel.DistributeDataParallel()` 코드 부분 하단에 위치시킨다. 

```python
# wrap model
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.gpu_to_work_on]
)

# # ImageNet pretraining 진행
swav_800ep_pretrain_Imagenet = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar')
model.load_state_dict(swav_800ep_pretrain_Imagenet)
```



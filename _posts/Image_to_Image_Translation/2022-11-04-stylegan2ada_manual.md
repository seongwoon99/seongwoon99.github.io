---
title: "StyleGAN2-ADA Train 메뉴얼"
excerpt: "StyleGAN2-ADA 모델의 github 코드를 통해 훈련을 진행하는 방법을 다룬다. 그리고 projector, generator 등 부가적인 모듈들을 다룬다."
categories:
  - Image_to_Image_Translation
tags:
  - stylegan
  - code
  - project
toc: true
toc_sticky: true
toc_label: "GITHUB BLOG JEKYLL"
toc_icon: "bookmark"
---

* Preparing Datasets
* Training new networks
* Generate Image
* Projecting images to latent space


## Preparing Datasets

* StyleGAN2-ADA를 train하기 전에 `.tfrecords` 형식으로 전처리를 진행한다. 
*  `jpg, png` 와 같은 이미지 파일 형식을 `.tfrecords` 형식의 확장자로 변환한다. 
* `dataset_too.py` 내부의 `create_from_images`를 호출한다. 


```python
# 실제 실행 코드
python dataset_tool.py create_from_images ~/tfrecords_folder/ ~/image_folder

# 설명용 코드 (실행 x)
# 인자는 2개로 tfrecords 파일을 저장할 빈 폴더와 이미지가 담긴 폴더 경로를 입력한다. 
python dataset_tool.py create_from_images  
    ~/tfrecords_folder/  # tfrecord 파일을 저장할 빈 폴더 경로 
    ~/image_folder/ # 이미지 폴더 경로
```



## Training new networks
* train 인자를 설정하고 훈련을 진행한다. 이때 경우에 따라서 경로를 미리 만들어야 할 수도 있다. (가급적이면 미리 만들어 놓는 것을 추천한다. )
* `--out_dir` : pkl 파일, 이미지 등 결과물이 저장된다. 
* `--gpus` : gpu 개수 설정 가능하다. colab에서 진행하는 경우 `--gpus=1`로 설정하고 진행해야 한다. 
* `--data` : tfrecord 경로를 지정한다. 
* `--resume` : 전이 학습 관련 옵션으로 특정 `pkl` 파일에 이어서 훈련을 진행하고 싶을 때 입력한다. (입력하지 않을 경우 해당 옵션을 제외하고 진행하면 된다.)
* `--kimg` : epoch과 비슷한 개념으로 `Discriminator` 가 한 번에 몇장의 이미지를 볼 지 설정하는 값이다. 해당 값에 도달할 때까지 훈련이 진행된다.  

```python
# 시험 코드로 `--dry-run` 옵션을 입력한다. 
# 모델의 parse 옵션들을 출력한다.
python train.py --outdir=~/training-runs --gpus=1 --data=~/datasets/custom --dry-run

# 본 train 코드
python train.py --outdir=~/results/ \
     --gpus=8\
     --data=~/tfrecords_folder \
     --resume=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl\
     --kimg=5000
```

## Generate Image

* `train.py` 결과로 생성된 `pkl` 파일로 이미지를 생성한다. 
* `seeds` : 이미지 개수 조절이 가능하다. 예를들어 10개의 이미지를 생성할 경우 `0-10` 으로 입력한다.
* `network` : `train.py` 의 결과인 `pkl` 파일을 입력한다. 

```python
python generate_direction.py --trunc=1 \
  --seeds=0-10 \
  --network=~/.pkl \
```

## Projecting images to latent space
* `projector.py` 를 통해 원하는 이미지에 대한 latent vector를 얻을 수 있다. 

```
python projector.py --outdir=out --target=targetimg.png \
    --network=~/.pkl
```

### Reference
  * [https://github.com/NVlabs/stylegan2-ada](https://github.com/NVlabs/stylegan2-ada)
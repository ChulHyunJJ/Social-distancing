# social-distancing

사회적 거리두기 감지 프로젝트입니다.

[monoloco](https://github.com/vita-epfl/monoloco) 를 기반으로 하여 수정하였습니다.

## 테스트 환경
```
macOS Catarina 10.15.7
python == 3.6.12
```

## 필수 설치 라이브러리

실행하기에 앞서 가상환경을 이용할 것을 권장합니다.

```
pip3 install torch==1.1.0
pip3 install torchvision==0.3.0
pip3 install openpifpaf==0.9.0
pip3 install pydub==0.24.1
```

설치가 끝났으면 Git 을 통해 복사합니다.<br><br>
`git clone https://github.com/ChulHyunJJ/social-distancing`

## 기능
- 실시간 사회적 거리두기 위반 감지 및 음향 출력
- 사람의 자세 및 시선 방향 인식
- *point of view* 로 부터의 거리 인식


## 비디오
현재 코드는 비디오를 입력으로 받는 것을 지원하지 않습니다.

FFMPEG 라이브러리를 통해 이미지로 변환하여 입력으로 사용할 수 있습니다.

Video --> Images:

`ffmpeg -i data/videos/park.mp4 -vf fps=6 data/images/%03d.png`

Images --> Video:

`ffmpeg -r 6 -f image2 -i data/output/%03d.png.front.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" processed_video.mp4`

## 데모 실행

1. 데모 이미지로 결과 출력하기 (기본 저장소 위치는 ./data/output/ 입니다.)

`ffmpeg -i data/videos/park.mp4 -vf fps=6 data/images/%03d.png` 를 실행하여 데모 이미지를 만듭니다.
```
python -m monoloco.run predict \
--social \
--glob "data/images/*.png" \
--networks monoloco \
--output_type front bird \
--model data/models/social.pkl \
-o data/output/ \
--z_max 30
```

2. 웹캠으로 실시간 비디오 불러오기
```
python -m monoloco.run predict \
--networks monoloco \
--output_type front bird \
--model data/models/social.pkl \
-o data/output/ \
--z_max 10 \
--scale 0.2 \
--webcam \
--social \
--show
```

옵션에 대한 자세한 설명은 `python3 -m monoloco.run predict --help` 을 통해 알아보거나 `'run.py'` 코드를 참조해주세요.

## Json 파일
정확한 거리 파악을 위해서는 Json 파일이 필요합니다.

Json 파일이 없더라도 상대적인 거리는 유의미하기 때문에 Json 파일이 없는 경우에도 제한된 환경에서 사용할 수 있습니다.

이미 존재하는 Json 파일을 불러오기 위해서는

`--json_dir <directory of json files>` 으로 불러올 수 있으며 없는 경우 pifpaf 를 통해 Json 파일을 생성할 수 있습니다.

```
python -m monoloco.run predict \
--model data/models/social.pkl \
--glob "data/images/*.png" \
--networks pifpaf \
--output_types json \
-o data/output/ \
--instance-threshold 0.4 
```

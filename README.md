# EmoLabel — 연구실 라벨링 도구

YouTube 숏츠에서 한 화자의 발화를 추출하고, 대사를 수정해 neutral / angry / sad / happy를 기록하는 로컬 웹 도구입니다. 중앙 서버 없이 각자 작업할 수 있습니다.

이번 배포본은 기존 Mac 앱의 복사본에 Linux ASR 경로를 추가했습니다. 원본 앱과 기존 데이터는 수정하지 않았습니다. 대상은 **Linux x86_64 또는 Apple Silicon Mac, Python 3.12**입니다. Linux ARM·Windows·Intel Mac은 이 배포본의 검증 대상이 아닙니다.

## 코드 가져오기

이 저장소는 비공개입니다. 저장소 접근 권한이 있는 GitHub 계정으로 인증한 뒤 복제하세요.

```bash
git clone https://github.com/yoonji4024/emolabel-youtube.git
cd emolabel-youtube
```

처음 작업하기 전에 [라벨링 가이드](docs/ANNOTATION_GUIDE.md)를 읽어주세요. [GPT 활용 가이드](docs/GPT_ASSISTANCE.md)는 선택 사항이며 GPT API 연결 없이도 앱을 실행할 수 있습니다.

수집한 101개 샘플과 모델 비교 결과는 이 저장소에 넣지 않았습니다. 데이터는 연구실의 접근 권한이 설정된 Google Drive로 별도 공유합니다. 다운로드 위치는 담당자에게 확인하세요. [데이터 설명](docs/DATASET_CARD.md)

## Linux 설치

Ubuntu/Debian에서 시스템 준비 (다른 배포판은 해당 패키지 관리자를 사용):

```bash
sudo apt update
sudo apt install ffmpeg git unzip
```

[uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 uv를 설치한 뒤, 이 폴더를 터미널에서 열고 실행합니다.

```bash
uv sync --locked --python 3.12
uv run python -m app.asr
uv run python -m app.gender
uv run python -m app.qc
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

같은 PC의 브라우저에서 http://127.0.0.1:8000 을 엽니다. 처음 모델을 받는 동안 시간이 걸리고 인터넷이 필요합니다. 다운로드 공간을 여유 있게 확보하세요. 실행 시간·용량은 PC 및 라이브러리 구성에 따라 달라지며, 기존 Mac의 속도를 Linux에 보장하지 않습니다.

Linux ASR은 `faster-whisper`, 기본 모델 `turbo`, CPU `int8`입니다. Apple Silicon에서는 기존 `mlx-community/whisper-large-v3-turbo`를 사용합니다. GPU는 기본적으로 요구하지 않습니다. [faster-whisper 공식 사용법](https://github.com/SYSTRAN/faster-whisper)

GPU를 쓰려면 먼저 CUDA/cuDNN 호환성을 확인한 후 별도로 설정하세요. CPU가 너무 느리면 `EMOLABEL_ASR_MODEL=small`로 시험할 수 있지만 전사 모델이 달라지므로 기록하고, 대사는 반드시 사람이 수정합니다. Mac의 `.env`에 있던 MLX 모델명을 Linux 설정에 그대로 복사하지 마세요.

## Mac 설치

Homebrew의 ffmpeg와 uv가 필요합니다. 설치되어 있다면 위의 `uv sync` 이후 명령을 동일하게 사용합니다. ffmpeg/ffprobe는 PATH에서 찾습니다. 필요한 경우 `.env.example`의 경로 설정을 사용하세요.

## 실행 확인 — 배포 전 마지막 단계

이 배포본은 분기/전사 형식 단위 테스트와 의존성 해석을 확인했습니다. **재훈님 Linux PC의 실제 영상 다운로드 → ASR → QC → 저장 전체 흐름은 아직 실행하지 않았습니다.**

먼저 짧은 샘플 하나로 다음을 확인한 뒤 연구실 인원에게 배포합니다.

1. `ffmpeg -version`, `ffprobe -version`이 실행됨.
2. 위의 모델 준비 명령이 성공함.
3. 영상 불러오기와 발화 추출이 됨.
4. 대사를 직접 수정하고 감정·본인 ID를 입력해 저장함.
5. `data/output/{clip_id}/` 안의 영상·WAV·annotation.json과 수정된 대사를 확인함.

단위 테스트 (모델 다운로드 없음):

```bash
uv run python -m unittest -v test_portability
```

8000 포트 사용 중이면 실행 중인 서버의 작업을 먼저 확인하고, 별도 시험은 `--port 8001`로 실행해 http://127.0.0.1:8001 에 접속하세요. 다른 사람이 쓰는 프로세스를 무작정 종료하지 않습니다. 인증이 없는 로컬 도구이므로 외부 공개를 위해 `--host 0.0.0.0`으로 바꾸지 마세요.

## 작업 및 저장

URL 불러오기 → 파형에서 발화 선택 → 구간 추출 → 원음과 전사 확인 → 감정·품질 메모·본인 ID → 저장 순서입니다. 전사와 QC는 보조 정보이며 감정을 자동 확정하지 않습니다.

```text
data/output/{clip_id}/
  clip.mp4
  audio_16k.wav
  prelabels.json      # 기계 초안: 생성된 경우
  annotation.json     # 사람 저장 완료 후 생성
```

정답은 `labels.emotion`과 `labels.transcript`입니다. `prelabels`는 기계 초안입니다. valence/arousal은 이번 단계에서 평가하지 않으며, 저장된 기본값 4를 평정값으로 분석하면 안 됩니다. Notes 기본값도 검수 완료 표시가 아니므로 매 샘플 직접 확인합니다.

진행상황·누락 점검:

```bash
uv run python tools/dataset_status.py --root data/output
```

## 공용 PC와 제출

한 번에 한 사람씩 작업합니다. 같은 브라우저는 이전 annotator ID를 기억하므로 교대 시 반드시 변경하세요. 각자의 별도 복사본에서 작업하거나 `.env`의 `EMOLABEL_DATA_DIR`에 서로 다른 **절대 경로**를 지정하세요. 모델 캐시만 공유하려면 `EMOLABEL_HF_HOME`을 공용 경로로 설정할 수 있지만, data/output은 분리합니다.

작업 종료 후 각자의 output을 `날짜_annotatorID` 폴더 아래 압축해 전달합니다. 병합 담당자는 같은 clip_id가 있으면 파일을 덮어쓰지 말고 대사·라벨 차이를 확인합니다. 서버로 자동 수집되지는 않습니다.

## 저장소에 포함되는 파일

이 비공개 저장소에는 앱 코드, 설치 의존성 잠금 파일, 실행 안내, 라벨링/GPT 활용 가이드만 포함합니다. 얼굴 수 QC에 필요한 기존 소형 YuNet 자산은 `app/models/`에 포함되어 있습니다. 공개 저장소로 바꾸기 전에는 코드·동봉된 서드파티 자산의 배포 조건을 확인하세요.

`.gitignore`는 `.env` 및 비밀 설정, data/, .venv/, 캐시, 영상·음성·ZIP, 다운로드한 모델 가중치를 제외합니다. 토큰·쿠키·수집 영상을 Git에 넣지 마세요. `.env.example`의 토큰은 비어 있으며, 토큰 없이도 기존 화자 수 QC 대체 경로를 사용합니다. 게이트 모델은 해당 모델의 이용 조건을 별도로 확인해야 합니다.

## 가이드

- [어노테이션 가이드](docs/ANNOTATION_GUIDE.md)
- [GPT 활용 가이드](docs/GPT_ASSISTANCE.md)
- [파일럿 데이터 설명](docs/DATASET_CARD.md)

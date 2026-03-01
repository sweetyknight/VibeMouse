# VibeMouse

按鼠标前侧键开始录音，再按一次结束录音并自动转文字。识别结果优先输入到当前焦点输入框；如果没有可编辑输入框，则自动写入剪切板。鼠标后侧键用于发送 Enter。

语音识别引擎使用 SenseVoice，支持两种后端：

- `funasr`（PyTorch）
- `funasr_onnx`（ONNXRuntime，Intel NPU 机器建议优先）

默认 `VIBEMOUSE_BACKEND=auto`，会自动选择更可用的后端。

## 功能

- 前侧键（默认 `x1`）：
  - 第一次按下：开始录音
  - 第二次按下：结束录音，触发 SenseVoice 转写
- 后侧键（默认 `x2`）：发送回车
- 转写输出策略：
  - 当前有可编辑输入焦点：直接键入
  - 否则：复制到剪切板
- 后端/设备策略：
  - 默认 `VIBEMOUSE_BACKEND=auto`
  - 默认 `VIBEMOUSE_DEVICE=cpu`（当前最稳）
  - 在 Intel NPU 场景下，`auto` 会优先尝试 `funasr_onnx`
  - 若后端或设备失败且 `VIBEMOUSE_FALLBACK_CPU=true`，自动降级 CPU

## 系统要求（Linux）

- Python 3.10+
- 音频录制支持（PortAudio / ALSA / PulseAudio）
- 全局输入监听权限（`/dev/input/event*`，推荐将用户加入 `input` 组）
- AT-SPI 可访问性（用于判断当前焦点是否为可编辑输入框）
- 本地 NPU 运行环境（推荐 Intel NPU + OpenVINO 可用）

建议先安装系统依赖（Debian/Ubuntu 示例）：

```bash
sudo apt update
sudo apt install -y python3-gi gir1.2-atspi-2.0 portaudio19-dev libsndfile1
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 运行

```bash
vibemouse
```

或：

```bash
python -m vibemouse.main
```

## 环境变量配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VIBEMOUSE_BACKEND` | `auto` | 转写后端：`auto` / `funasr` / `funasr_onnx` |
| `VIBEMOUSE_MODEL` | `iic/SenseVoiceSmall` | 模型名/路径（`funasr_onnx` 推荐 `iic/SenseVoiceSmall-onnx`） |
| `VIBEMOUSE_DEVICE` | `cpu` | 设备偏好（默认 CPU 稳定模式；可手工设为 `npu:0` / `cuda:0`） |
| `VIBEMOUSE_FALLBACK_CPU` | `true` | 设备失败时是否自动退回 CPU |
| `VIBEMOUSE_LANGUAGE` | `auto` | 语言（`auto`/`zh`/`en`/`yue`/`ja`/`ko`） |
| `VIBEMOUSE_USE_ITN` | `true` | 是否启用 ITN |
| `VIBEMOUSE_ENABLE_VAD` | `true` | 是否启用 `fsmn-vad` |
| `VIBEMOUSE_VAD_MAX_SEGMENT_MS` | `30000` | VAD 单段最大毫秒数 |
| `VIBEMOUSE_MERGE_VAD` | `true` | 是否合并 VAD 碎片 |
| `VIBEMOUSE_MERGE_LENGTH_S` | `15` | 合并长度（秒） |
| `VIBEMOUSE_SAMPLE_RATE` | `16000` | 录音采样率 |
| `VIBEMOUSE_CHANNELS` | `1` | 录音声道 |
| `VIBEMOUSE_DTYPE` | `float32` | 录音数据类型 |
| `VIBEMOUSE_FRONT_BUTTON` | `x1` | 前侧键（`x1` 或 `x2`） |
| `VIBEMOUSE_REAR_BUTTON` | `x2` | 后侧键（`x1` 或 `x2`） |
| `VIBEMOUSE_TEMP_DIR` | 系统临时目录下 `vibemouse` | 临时录音目录 |

示例：

```bash
export VIBEMOUSE_DEVICE=npu:0
export VIBEMOUSE_LANGUAGE=auto
vibemouse
```

Intel NPU 推荐配置（自动优先 ONNX 后端）：

```bash
export VIBEMOUSE_BACKEND=auto
export VIBEMOUSE_MODEL=iic/SenseVoiceSmall-onnx
export VIBEMOUSE_DEVICE=npu:0
vibemouse
```

稳定推荐（先用这个）：

```bash
export VIBEMOUSE_BACKEND=auto
export VIBEMOUSE_DEVICE=cpu
vibemouse
```

## 权限说明

如果程序无法监听侧键，通常是输入设备权限问题。可将当前用户加入 `input` 组并重新登录：

```bash
sudo usermod -aG input $USER
```

## SenseVoice / NPU 说明

项目内部支持两条推理链路：

1) `funasr`：

- `AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device="...")`
- `model.generate(...)`
- `rich_transcription_postprocess(...)`

2) `funasr_onnx`：

- `SenseVoiceSmall(model_dir="iic/SenseVoiceSmall-onnx", quantize=True, ...)`
- `model(audio_path, language=..., textnorm=...)`
- `rich_transcription_postprocess(...)`

> 注意：当前 `funasr_onnx` 官方 Python 实现主要支持 CPU / CUDA provider；Intel NPU 设备虽可被 OpenVINO 识别，但直接对 SenseVoice ONNX 编译到 NPU 可能因动态形状限制失败。此时会自动回退 CPU。若要强制 NPU，需要额外的静态图改造/编译流程。

如果你的 NPU 栈需要额外 runtime，请先按本机驱动文档安装好再运行。

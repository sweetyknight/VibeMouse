# VibeMouse

**鼠标侧键语音输入工具 — 支持 Windows 和 Linux。**

VibeMouse 把鼠标侧键变成语音输入快捷键：

- 按住侧键开始录音，VAD 智能断句 + 离线高精度识别，整句文字即时出现
- 松开侧键结束，识别结果直接输入到当前焦点窗口
- 支持**长按模式**和**开关模式**，可在托盘菜单实时切换
- 另一个侧键发送 Enter 提交
- Windows 下带系统托盘图标，状态一目了然

适合在 ChatGPT / Claude / IDE 里用语音写提示词或代码注释。

---

## 功能

- **VAD + 离线高精度识别** — Silero VAD 智能断句 + FireRedASR 离线识别，整句文字一次出现，准确率高
- **音频预缓冲** — 录音前 0.5 秒预缓冲，不丢第一个音节
- **跨平台** — Windows（系统托盘 + Win32 hook）/ Linux（evdev + Atspi）
- **绕过输入法** — Windows 上通过 `SendInput` + `KEYEVENTF_UNICODE` 直接注入 Unicode，不触发 IME
- **单实例保护** — 防止重复启动创建多个托盘图标
- **低资源占用** — 批量音频处理、队列限容、图标缓存等优化
- **录音模式切换** — 长按模式（按住录音）/ 开关模式（点击切换），托盘菜单实时切换

---

## 支持平台

| 平台                | 侧键监听              | 文字输出          | 托盘图标 |
|---------------------|-----------------------|-------------------|----------|
| Windows 10/11       | Win32 低级鼠标 Hook   | SendInput Unicode | pystray  |
| Linux (X11/Wayland) | evdev                 | pynput / Atspi    | —        |

Python 3.10+

---

## 快速开始

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
vibemouse
```

首次启动会自动下载 Silero VAD 模型和 FireRedASR 离线识别模型（共约 1.5 GB），之后启动秒开。
启动后系统托盘出现绿色圆点图标，录音时变红，识别时变橙。

日志文件位于 `~/.cache/vibemouse/vibemouse.log`，EXE 模式下所有输出自动写入该文件。

### Linux (Ubuntu/Debian)

```bash
sudo apt install -y python3-gi gir1.2-atspi-2.0 portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
vibemouse
```

> 如果侧键监听不到，把用户加到 `input` 组：`sudo usermod -aG input $USER`，然后重新登录。

---

## 默认按键映射

| 侧键 | 功能 |
| --- | --- |
| `x1`（前侧键） | 长按模式：按住录音，松开停止；开关模式：点击开始/停止 |
| `x2`（后侧键） | 发送 Enter |

如果你的鼠标按键相反：

```bash
export VIBEMOUSE_FRONT_BUTTON=x2
export VIBEMOUSE_REAR_BUTTON=x1
vibemouse
```

---

## 工作流程

### 长按模式（默认）

1. **按住** 前侧键，开始录音（VAD 实时断句，识别完成后整句出现）
2. **松开** 前侧键，停止录音，最终文字保留在输入框中
3. 按后侧键发送 Enter 提交

### 开关模式

1. **点击** 前侧键，开始录音（托盘图标变红）
2. **再次点击** 前侧键，停止录音，文字输出到焦点窗口
3. 按后侧键发送 Enter 提交

> 在托盘菜单中勾选「Toggle mode (click to record)」即可切换到开关模式，取消勾选回到长按模式。

---

## 配置（环境变量）

### 核心配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VIBEMOUSE_FRONT_BUTTON` | `x1` | 语音侧键（`x1` / `x2`） |
| `VIBEMOUSE_REAR_BUTTON` | `x2` | Enter 侧键（`x1` / `x2`） |
| `VIBEMOUSE_BUTTON_DEBOUNCE_MS` | `150` | 侧键去抖窗口（毫秒） |
| `VIBEMOUSE_ENTER_MODE` | `enter` | 提交模式：`enter` / `ctrl_enter` / `shift_enter` / `none` |
| `VIBEMOUSE_RECORDING_MODE` | `hold` | 录音模式：`hold`（长按）/ `toggle`（开关） |
| `VIBEMOUSE_AUTO_PASTE` | `true` | 是否自动粘贴 |

### 模型配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VIBEMOUSE_SHERPA_MODEL_DIR` | `~/.cache/vibemouse/models` | 模型存储目录 |
| `VIBEMOUSE_SHERPA_NUM_THREADS` | `2` | 推理线程数 |
| `VIBEMOUSE_OFFLINE_MODEL_NAME` | `sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16` | 离线识别模型名称（中英双语） |

### VAD 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VIBEMOUSE_VAD_MIN_SILENCE_DURATION` | `0.25` | 最小静音时长（秒），用于断句 |
| `VIBEMOUSE_VAD_MIN_SPEECH_DURATION` | `0.25` | 最小语音时长（秒），过短片段被忽略 |
| `VIBEMOUSE_VAD_THRESHOLD` | `0.5` | VAD 置信度阈值（0–1） |

### 音频配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VIBEMOUSE_SAMPLE_RATE` | `16000` | 采样率（Hz） |
| `VIBEMOUSE_CHANNELS` | `1` | 声道数 |
| `VIBEMOUSE_DTYPE` | `float32` | 音频数据类型 |
| `VIBEMOUSE_PRE_BUFFER_SECONDS` | `0.5` | 录音前预缓冲时长（秒），防止首音节丢失 |

### Windows 专属

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VIBEMOUSE_NO_TRAY` | `false` | 设为 `true` 禁用托盘模式（改为控制台模式） |

---

## 常见问题

### Linux: 侧键监听不到

通常是输入设备权限问题：

```bash
sudo usermod -aG input $USER
# 重新登录生效
```

### 后侧键 Enter 不稳定

加大去抖或切换提交组合键：

```bash
export VIBEMOUSE_ENTER_MODE=ctrl_enter
export VIBEMOUSE_BUTTON_DEBOUNCE_MS=220
```

Hyprland 用户可以用合成器级绑定替代：

```ini
# ~/.config/hypr/UserConfigs/UserKeybinds.conf
bind = , mouse:276, sendshortcut, , Return, activewindow
```

```bash
export VIBEMOUSE_ENTER_MODE=none
```

### 能录音但识别为空

检查麦克风增益/输入源，确认录到的不是静音。

---

## 构建 Windows EXE（可选）

```bash
# 方式一：使用 build.bat 一键构建
build.bat

# 方式二：手动构建
pip install "pyinstaller>=6.0"
python scripts/build_exe.py
# 输出: dist/VibeMouse.exe
```

---

## 项目结构

```text
vibemouse/
  main.py              # 入口（托盘/控制台模式选择、单实例锁）
  app.py               # 主流程编排（录音↔识别↔输出）
  audio.py             # 音频录制（sounddevice）+ 预缓冲
  transcriber.py       # 类型定义（StreamingResult / AudioFrame）
  vad_transcriber.py   # VAD + 离线识别（Silero VAD + FireRedASR）
  model_manager.py     # 模型下载与路径管理
  streaming_output.py  # 流式文字输出（Win32 Unicode / pynput）
  output.py            # Enter 键发送（Atspi / Hyprland / pynput）
  mouse_listener.py    # 侧键监听（Win32 Hook / evdev / pynput）
  config.py            # 环境变量配置
  tray.py              # Windows 系统托盘图标
  __main__.py          # python -m vibemouse 支持
```

---

## 后台常驻运行（Linux）

```bash
# tmux
tmux new -d -s vibemouse "source .venv/bin/activate && vibemouse"

# 或 systemd user service
```

Windows 下可通过托盘菜单勾选「Auto-start with Windows」实现开机自启。
托盘菜单还可实时切换录音模式（长按 / 开关）。

---

## License

VibeMouse 项目源码采用 Apache-2.0 许可证，详见 `LICENSE`。

第三方依赖与模型资产声明见 `THIRD_PARTY_NOTICES.md`。

在分发二进制或打包模型前，请复核 LGPL 依赖（`pynput`、`PyGObject`）
的合规要求，并确认你实际使用的模型许可证。

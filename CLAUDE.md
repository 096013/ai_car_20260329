# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI 自走車控制系統，以 Raspberry Pi 為基礎，透過瀏覽器介面進行手動/自動駕駛切換。

## Running the App

```bash
python app.py
```

Runs on `http://0.0.0.0:5000` by default. Requires Raspberry Pi hardware for full functionality; non-RPi environments use dummy camera and GPIO stubs automatically.

## Architecture

### `app.py` — Flask Web Server & AI Logic
- **`PIDController`** — 計算車道偏移的 PID 輸出，用於自動轉向
- **`DummyCamera`** — 非 RPi 環境的假攝影機，產生黑色畫面
- **Camera thread** — 背景執行緒持續從 `Picamera2` 或 `DummyCamera` 擷取畫面
- **Lane detection** — HSV 色彩空間偵測黃/藍/白車道線，計算中線偏移
- **YOLO integration** — 若存在 `yolo_model.pt`，偵測紅燈並暫停自動駕駛
- **Auto-drive loop** — 背景執行緒讀取車道偏移，透過 `robot.drive()` 控制馬達
- **`/status` endpoint** — 前端每秒輪詢，回傳 AI 狀態、偏移量、物件列表等

### `motor_control.py` — 硬體抽象層
- **`PCA9685`** — I2C PWM 控制器驅動（位址 `0x40`，頻率 50 Hz）
- **`RobotControl`** — 四輪馬達控制，支援 `arc`（差速轉彎）與 `spin`（原地旋轉）兩種轉彎模式
  - Motor A/C = 左側，Motor B/D = 右側
  - Motor D 使用 GPIO 25/24 控制方向（`motorD1`/`motorD2`）
  - 可透過 `LEFT_MOTOR_SCALE` / `RIGHT_MOTOR_SCALE` 校正左右輪速差

### Frontend (`templates/index.html`, `static/`)
- 純 HTML/CSS/JS，無框架
- 手動模式：方向鍵或螢幕按鈕觸發 `POST /move`
- 自動模式：切換後由後端 AI loop 控制，前端只顯示狀態

## Key Configuration Constants (in `app.py`)

| 常數 | 說明 |
|------|------|
| `YOLO_MODEL_PATH` | YOLO 模型路徑（`yolo_model.pt`） |
| `TRAFFIC_LIGHT_LABELS` | 視為紅燈的 YOLO 標籤集合 |
| `LANE_COLOR_RANGES` | HSV 車道顏色範圍（yellow/blue/white） |
| `PIDController(kp, ki, kd)` | 預設 kp=0.35, ki=0.0, kd=0.18 |

## Hardware Dependencies

- `picamera2` — Raspberry Pi 攝影機（缺少時自動降級為 `DummyCamera`）
- `gpiozero` — GPIO 控制（缺少時使用 stub `OutputDevice`）
- `smbus` — I2C 通訊（缺少時使用 `DummySMBus`）
- `ultralytics` — YOLO 推論（缺少時跳過物件偵測）

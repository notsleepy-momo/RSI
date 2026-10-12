# ALFWorld 单题（Windows + Docker + Ollama）

本目录只运行官方 ALFWorld 的 TextWorld 文本环境，不安装 `alfworld[full]`，因此不需要 AI2-THOR、OpenGL、摄像头或 GPU。Windows 只负责 Docker Desktop 和 Ollama；ALFWorld、TextWorld、Jericho 在 Linux 容器内运行。

## 前置条件

1. Windows Docker Desktop 已启动（Linux/WSL2 backend）。
2. Windows Ollama 已启动，并且存在模型：

   ```powershell
   ollama pull qwen3.5:4b
   ```

3. Ollama 至少能够在 Windows 本机通过 `http://127.0.0.1:11434` 访问。

本机现有模型实际位于 `C:\Users\ASUS\.ollama\models`。如果 `ollama list` 为空或 Ollama 尚未运行，请先在一个独立 PowerShell 窗口执行并保持窗口开启：

```powershell
cd E:\科研\ppt\25262\rsi\benchmark\ALFworld
.\start_ollama_localhost.ps1
```

## 一键单题

在 PowerShell 中执行：

```powershell
cd E:\科研\ppt\25262\rsi\benchmark\ALFworld
$env:DOCKER_BUILDKIT = "0"     # Docker Desktop 偶发 BuildKit EOF 时更稳定
.\run_one_ollama.ps1 -Model qwen3.5:4b -TaskIndex 0 -MaxSteps 1
```

第一次会构建镜像并下载官方文本数据；数据会缓存到 `data`，后续运行不会重复下载。结果写入 `results`。默认采用安全的 localhost bridge：容器生成官方 observation 和 admissible actions，PowerShell 在 Windows localhost 调 Ollama，再把模型动作交回同一官方任务。这不要求把 Ollama 暴露到 `0.0.0.0`。

如果你本来就把 Ollama 配置为可从 Docker 访问，可加 `-DirectContainerOllama`，直接使用 `http://host.docker.internal:11434`；该模式也支持多步。

## 分步运行

```powershell
$docker = "D:\software\Docker\resources\bin\docker.exe"
docker build -f Dockerfile.ollama -t alfworld-ollama:qwen35 .
docker run --rm --mount "type=bind,source=$((Resolve-Path .\data).Path),target=/opt/alfworld_data" --entrypoint python alfworld-ollama:qwen35 /opt/alfworld_project/download_alfworld_data.py --data-dir /opt/alfworld_data
docker run --rm --mount "type=bind,source=$((Resolve-Path .\data).Path),target=/opt/alfworld_data" --mount "type=bind,source=$((Resolve-Path .\results).Path),target=/results" alfworld-ollama:qwen35 --model qwen3.5:4b --task-index 0 --max-steps 1 --output /results/one.json
```

## 实验口径

该实验验证的是“官方 ALFWorld 任务加载 → 文本 observation/admissible actions → Ollama Qwen3.5-4B HTTP 调用 → 动作提交 → TextWorld 反馈”的端到端连通性。`MaxSteps=1` 只提交一步动作，`won=false` 不表示安装失败；单题也不代表完整 ALFWorld benchmark 分数。

视觉 THOR 模式需要额外的 `alfworld[full]`、AI2-THOR 2.1.0、渲染/图形依赖和模型权重，不纳入当前 Windows/Ollama 文本实验。

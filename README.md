# 4 人 Hermes 内部 LLM / RunPod Serverless

本目录只提供部署文件，未执行构建或部署。核验日期：2026-11-17。

## 构建前必须解决的上游差异

- 指定的 [Docker Hub tags API](https://hub.docker.com/v2/repositories/runners/runpod-worker-cuda/tags?page_size=100) 实测 HTTP 404，不能确认该镜像的公开可用性、官方归属或最新 cu12.x tag。因此 Dockerfile 的 `RUNPOD_CUDA_TAG` 是必填 build arg，没有虚构默认 tag。需要取得真实 tag，并验证镜像内 `/handler/main.py` 会启动/导入 `/server.py`，将请求交给端口 8000；当前不能声称镜像可直接构建部署。
- [llama.cpp latest API](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest) 返回 `v1.0.0`，仅含 `nightly-tag.txt`，没有 `linux-amd64-Cuda.tar.gz`。Dockerfile 固定到该正式版明确链接的 [b12359](https://github.com/ggml-org/llama.cpp/releases/tag/b12359)，下载真实的 `llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz` 及配套 `cudart-...`，按 API SHA256 校验。它是 latest 正式版关联的构建，不是最新 nightly；本次也观察到更新的 nightly 发布。
- 资产保持目录结构，不使用 `--strip-components`。未下载压缩包检查目录，故 `LLAMA_BIN_DIR=/opt/llama.cpp` 作为递归搜索根；启动日志输出发现的真实二进制路径。发现多个二进制时，必须设置 `LLAMA_SERVER_BIN`，不会猜测。配套 `.so` 目录自动加入子进程 `LD_LIBRARY_PATH`。
- CUDA 资产为 **12.8**：基础镜像/宿主驱动必须满足该运行时及 RTX PRO 6000 Blackwell 的要求，不能把任意 cu12.x tag 当成已验证兼容。没有进行 GPU 实测。

## Worker、卷和模型

目标选择：`PRO_6000_96`，RTX PRO 6000 96 GB，`SECURE + COMMUNITY`，location `EUC`；这些是部署目标，本次未查询 RunPod API 验证库存或资源保证。

逻辑挂载约定：**卷根挂载 `/data`，卷内目录为 `models/`**，最终路径 `/data/models/<filename>`。不要把已经含 `models/` 的卷根再次挂到 `/data/models`，否则会形成双层目录。

[RunPod 当前 Serverless Global Volume 文档](https://docs.runpod.io/storage/globalvolume/globalvolume-serverless) 写明实际挂载点为 `/runpod-volume`。若控制台不能指定 `/data`，保持官方挂载，并在卷根保留 `models/`；server.py 在 `/data/...` 文件不存在时自动尝试 `/runpod-volume/...`，无需复制模型。`VOLUME_FALLBACK_ROOT` 可改此根路径，设空可关闭回退。若模型直接放在卷根，显式把 `MODEL_FILE`、`MMPROJ_FILE` 改为 `/runpod-volume/<filename>`；只有部署平台实际支持自定义挂载时才用 `/data`。

以下数字来自 [Hugging Face tree API](https://huggingface.co/api/models/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/tree/main?recursive=true) 的 `size`，不是估计值。GB=10^9 bytes，GiB=2^30 bytes。

| 文件（位于模型目录） | bytes |
| --- | ---: |
| Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00001-of-00003.gguf | 46,663,299,456 |
| Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00002-of-00003.gguf | 46,601,152,320 |
| Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00003-of-00003.gguf | 6,982,791,168 |
| Qwen3.8-Flash-Next-Uncensored-IQ3_M-00001-of-00002.gguf | 44,706,357,824 |
| Qwen3.8-Flash-Next-Uncensored-IQ3_M-00002-of-00002.gguf | 44,834,460,608 |
| mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf | 907,543,296 |

IQ4_XS-MTP 合计 **100,247,242,944 bytes（100.247 GB，93.363 GiB）**；IQ3_M 合计 **89,540,818,432 bytes（89.541 GB，83.391 GiB）**。视觉塔另计；两套模型加视觉塔共 **190,695,604,672 bytes（190.696 GB）**，卷容量需留余量。仅传第一片给 llama.cpp；程序先检查所有分片存在，再让 llama.cpp 自动发现其余分片。不会从网络自动下载权重。

IQ4_XS-MTP 约 100 GB 文件默认走 llama.cpp mmap，不加 `--no-mmap`。按本方案，选择**实际分配给 worker 的主机 RAM >150 GB**的 PRO 6000 EUC 机型，才满足这里的主机内存预算；不能仅凭 GPU 池名保证每台 worker 的 RAM。首次页缓存全冷时按 **3–6 分钟**预留冷启动，这是部署预算，尚非本项目实测；Global Volume 吞吐、宿主页缓存和资源争用会改变实际时间。

主机 RAM 与 VRAM 分别预算：93.363 GiB 权重之外还要容纳视觉塔、KV、计算缓冲、MTP 状态。默认 `-ngl 99`、262K context、q8 KV 在 96 GB 显卡上**尚未验证能同时容纳**，主机 RAM 充足不能解决全 GPU offload 的 OOM。若 OOM，先降低 `N_GPU_LAYERS` 或 `CTX_SIZE`，或切 IQ3_M；不要把 mmap 等同于自动 CPU offload。

## 启动参数与并发

默认命令包含：`--host 127.0.0.1 --port 8080 -ngl 99 --flash-attn on -c 262144 --parallel 4 --cache-type-k q8_0 --cache-type-v q8_0 --jinja --timeout 1200 --alias team-hermes`，并携带默认模型和视觉塔路径。`--jinja` 用于 Hermes 工具调用。

| 环境变量 | 默认 / 用途 |
| --- | --- |
| `MODEL_FILE` | `/data/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00001-of-00003.gguf` |
| `MMPROJ_FILE` | `/data/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`；空值禁用视觉塔 |
| `MODEL_ALIAS` | `team-hermes`；团队配置固定使用此模型 ID |
| `N_GPU_LAYERS` / `FLASH_ATTN` | `99` / `on` |
| `CTX_SIZE` / `PARALLEL` | `262144` / `4` |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` / `q8_0` |
| `SPEC_MTP` / `SPEC_DRAFT_MAX` | `0` / `4`；仅 `SPEC_MTP=1` 启用 MTP |
| `CHAT_TEMPLATE_KWARGS` | 空；默认不添加全局 thinking 参数，由 Hermes 每次请求控制 |
| `REQUEST_TIMEOUT` | `1200` 秒；httpx 各网络阶段超时及 llama-server read/write timeout |
| `LLAMA_STARTUP_TIMEOUT` | `1200` 秒；未首次就绪超过此值则终止并重启 |
| `RESTART_INITIAL_SECONDS` / `RESTART_MAX_SECONDS` | `2` / `60`，指数退避；稳定运行满 60 秒后重置 |
| `LLAMA_EXTRA_ARGS` | 空；shlex 拆分的额外参数，不通过 shell 执行；不要在此重复 MTP、host、port 参数 |

总 context 为 **262,144 tokens = 4 slots × 65,536 tokens**（约 4 × 65K = 262K），并非每位成员 262K。每个 slot 的系统提示、历史、工具定义、图片及输出共用预算；Hermes 的 `context_length` 应填 `65536`。将 worker 的 RunPod 请求并发上限也设为 4；4 人还可能产生压缩、标题等后台请求，超出时会排队。只启动 **1 个 ASGI 进程**，不要开多 uvicorn worker，否则会争抢 8080 并重复加载模型。

启用 `SPEC_MTP=1` 时追加 `--spec-type draft-mtp --spec-draft-max 4`。若子进程在首次就绪前失败，监控线程立即去掉这两组参数重试一次；本 worker 生命周期后续按不带 MTP 的配置退避重启。已就绪后的崩溃先保留 MTP 重启，若再次启动失败才触发这一次回退。参数兼容回退不能修复模型架构不受支持或 OOM。

切换小模型：设置 `MODEL_FILE=/data/models/Qwen3.8-Flash-Next-Uncensored-IQ3_M-00001-of-00002.gguf`，同时设 `SPEC_MTP=0`，重新启动 worker；`MODEL_ALIAS=team-hermes` 保持团队配置稳定。

## HTTP 契约和对外地址

- `GET /`：200；`GET /health`：已就绪为 200 `{"message":"ready"}`，未就绪仍为 200 `{"message":"booting"}`。这是指定模板契约，HTTP 200 本身不能当作模型已加载；推理未就绪时返回 503，需重试。
- `POST /handle_request`：接收完整 OpenAI chat body，不再拆 `input`，代理到内部 `/v1/chat/completions`。
- `GET /v1/models` 与 `POST /v1/chat/completions`、`/v1/completions`、`/v1/embeddings`：保留请求 body、查询参数、上游状态码与响应。流式使用 httpx + `StreamingResponse(text/event-stream)` 透传原始 SSE 字节及上游 `data: [DONE]`；断流不会伪造成功结束。
- `GET /ws`、`GET /get_status`：200 stub，不代表真正支持 WebSocket。embeddings 路由透传也不保证当前生成模型/启动模式支持 embedding。

[RunPod 官方集成文档](https://docs.runpod.io/integrations/overview) 确认其 OpenAI 兼容前缀为：

```text
base_url: https://api.runpod.ai/v2/{endpoint_id}/openai/v1
chat:     https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions
```

该文档明确覆盖 vLLM/SGLang 等兼容 worker；本指定模板的转发实现不可取得，**不能仅凭实现本地 `/v1/*` 就保证网关会路由到它**。需部署时验证模板将 OpenAI 请求送往 `/handle_request`、将 SSE 正确返回，并支持所需 `/models` 请求。这里未调用该端点，也未做网关联调。

按团队约定，客户端 `Authorization: Bearer <该 endpoint 的 user token>`；当前 [官方文档](https://docs.runpod.io/serverless/endpoints/send-requests) 示例名称是 `RUNPOD_API_KEY`。特定 user token 是否具备该 endpoint 的推理权限待管理员验证，不要把 endpoint_id 或控制台登录会话当作 API key。

如果 OpenAI 网关未接通，标准队列 API 的调用流程（需要模板确实实现队列 handler）是：

```text
POST https://api.runpod.ai/v2/{endpoint_id}/run
Authorization: Bearer <RUNPOD_API_KEY 或已验证 user token>
body: {"input":{"model":"team-hermes","messages":[{"role":"user","content":"你好"}],"stream":true},"policy":{"executionTimeout":1200000}}
响应取得 job_id；随后轮询以下二者之一：
GET https://api.runpod.ai/v2/{endpoint_id}/stream/{job_id}  （增量结果）
GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}  （任务状态/结果）
```

`/stream/{job_id}` 的队列结果格式不同于 OpenAI SSE，不能直接作为 Hermes `base_url`；`POST /run/{id}` 不是标准提交路径。也可按 [Load balancing 官方指南](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod/golden-paths/14-load-balancing-endpoint.md) 暴露本 HTTP 服务，使用 `https://{endpoint_id}.api.runpod.ai/v1` 作为 base_url，但须另行核实模板是否适用。

冷启动、任务执行超时和客户端无进展超时是不同配置：RunPod execution timeout 至少 1200 秒，初始化等待需覆盖冷启动；Hermes 请求超时也应允许 1200 秒，客户端若在首 token 前先超时，单独增大服务端超时无效。4 人工作时段可保留 1 个 active worker 来减少冷启动，产生相应空闲费用。

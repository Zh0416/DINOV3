# 两张固定训练 RGB 的 CLIP 滑窗区域诊断

## 已收到的云端证据

用户在魔搭执行了脚本12，本地未连接云端。固定 train RGB 是 `6ad03c4a...` 和
`03912ddf...`，均 512×512，InternVL 视觉尺寸/force_image_size 均为 448，8 次调用各 1 patch。
512² 未达到当前规则选择 2×2 切块的面积条件，单 patch 符合代码。缩小可能损失小目标，
但目前没有切块错误证据，也不能将缩小认定为主要根因。

| 层次 | 真实结果 | 判断 |
|---|---|---|
| 解析/推理 | 8 次成功，无截断 | 本次不能归因于历史 JSON 问题 |
| 第一张 presence | 原图/翻转均负 | 未知是否正确 |
| 第一张 grid | 原图/翻转均16格全正 | 与 presence 矛盾；一致也可能稳定退化 |
| 第二张 grid | 原图/翻转均16格全负 | 无标签时不能断言对错 |
| 第一张教师/RGB交集 | 前景98、背景0 | 这次直接阻断是缺背景种子，而非完全没有正种子 |
| 第二张 RGB 先验 | 前景3538/4096、背景0 | RGB硬门槛进一步限制双类支持，尚不能直接取消 |
| 原型/DINO过滤 | 因双类种子不足未执行 | 最终伪监督覆盖 null（未测），不能归因于 DINO/Decoder |

本次实际配置 FP16+8bit（请求auto），presence合计16.810秒，grid97.169秒，加载42.213秒，
总计156.686秒。用户之前BF16/不量化仍反转属于另一轮，不能混为本轮设置。
保留冻结 DINOv3 学生、轻量头、冲突弃权、双视图支持、训练划分和标签防火墙；
先暂停 InternVL 网格到像素监督，不扩大到全量教师，不改 Decoder，不调阈值凑种子。

## 最小实现

独立脚本13只运行一个冻结 `openai/clip-vit-base-patch16`，远端版本固定为
`57c216476eefef5ab752ec549e440a49ae4ae5f3`。配置 `configs/clip_window_probe_v1.yaml`
固定光伏文本及水面、暗屋顶、阴影、温室、植被、道路、裸地竞争文本。
同一 CLIP 对齐空间中，每条文本归一化、组内平均再归一化，图像嵌入归一化后计算余弦。
不将 DINO 向量与文本直接比较，不调用 InternVL/SAM/DINO特征提取。

512×512上使用256窗、128步长，每视图9窗，两图原图/翻转共36裁窗，batch8、FP32。
窗口包含边缘，非整除及奇数尺寸也保证翻转坐标封闭。前/背景均需相似度>=0.20、
领先相反概念margin>=0.03，否则unknown；这是事先固定的诊断起点，不是校准概率或已验证阈值。

**窗口存在性不等于像素占用。** 窗口广播只用于区域覆盖统计和可视化；重叠正负意见保留unknown，
unknown不投负票，双视图共同支持同一状态才进入共识。未检出不自动转全图背景。
没有生成训练伪标签、原型或 Adapter 语义输入，不是 ProxyCLIP 复现或已验证的密集分割教师。
逐窗记录坐标、余弦、margin、状态，记录提示词、RGB哈希、模型来源、实际精度、处理器和耗时。
四栏PNG展示RGB/原图区域/还原翻转区域/共识。目录名不参与类别或选图；只消费固定 manifest。
本地模型目录记录路径与配置，但不宣称已核验其权重字节与官方固定版本一致。

## 云端命令

先拉取此次代码，保留已有服务器路径。无需重跑 prepare、features、InternVL 或 train。
模型需要标准 HF CLIP 目录，不能直接放OpenAI `.pt` 或中文CLIP权重。
先检查已有环境 `python -c "import torch; print(torch.__version__)"`：官方固定版本仅提供
`pytorch_model.bin`，当前Transformers安全加载此格式要求PyTorch>=2.6；旧环境不要盲目覆盖升级。
如果还没有模型且服务器能访问HF，用官方API下载一次（约600MB）：

```bash
cd /mnt/workspace/DINOV3
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch16', revision='57c216476eefef5ab752ec549e440a49ae4ae5f3', local_dir='/mnt/workspace/models/clip-vit-base-patch16', allow_patterns=['*.json','merges.txt','vocab.json','pytorch_model.bin'])"
```

已核对官方文件列表；没有核验到可用魔搭同权重镜像，不提供猜测模型ID。
若下载失败，先解决权重获取，不将其他权重悄然当同一协议。已有标准目录时无需下载。

### 云端无法联网时：使用经校验的离线包

用户反馈官方源不可达、镜像连接超时；这些错误发生在下载阶段，尚未进入 CLIP 推理。
无需因此更换模型或重装现有环境。可在能联网的电脑准备同一固定版本的8个模型文件，
上传离线包；不要打包下载缓存、锁文件或未完成文件。官方固定权重信息为：

```text
revision: 57c216476eefef5ab752ec549e440a49ae4ae5f3
pytorch_model.bin bytes: 598641023
SHA256: ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0
```

上述字节数与 SHA-256 来自官方该版本的 LFS 元数据，不是根据本地文件自行假定。
本次离线包名为 `clip-vit-base-patch16-offline.tar.gz`，顶层目录为
`clip-vit-base-patch16-offline`，含原始模型文件、`OFFLINE_MODEL_SOURCE.json` 和 `SHA256SUMS`。
将包上传到 `/mnt/workspace/models/` 后执行：

```bash
cd /mnt/workspace/models
tar -xzf clip-vit-base-patch16-offline.tar.gz
cd clip-vit-base-patch16-offline
sha256sum -c SHA256SUMS
```

仅当全部显示 `OK` 后继续。该目录独立于先前的下载目录，避免混用未完成文件。
随后在同一终端设置本地路径与离线模式；保留已有 DINO/InternVL 路径：

```bash
export CLIP_MODEL_PATH=/mnt/workspace/models/clip-vit-base-patch16-offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

再运行下方两图诊断命令，入口会保留已设置的 `CLIP_MODEL_PATH`。
下载、校验或合成 RGB 接口通过，都不代表真实光伏语义已验证。

```bash
cd /mnt/workspace/DINOV3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/mnt/workspace/models/clip-vit-base-patch16}"
python -u scripts/13_clip_window_probe.py \
  --config configs/clip_window_probe_v1.yaml \
  --manifest runs/teacher_protocol_probe_v4_001/rgb_manifest.json \
  --compare-report runs/teacher_protocol_probe_v4_001/teacher_protocol_probe.json \
  --output-root runs/clip_window_probe_v1_001
```

注意用上次**两图探针manifest**，不是四图smoke manifest。
完整清单必须只有1--2张train RGB，内容哈希匹配；旧报告的ID、顺序、哈希也必须一致。
新目录逐视图保存报告并打印进度，异常保留已完成结果。无需更改已有 DINO/InternVL 路径。

## 解读与停止条件

- `records[].pair`：三态一致率、共同有效覆盖/一致率、区域正证据IoU、冲突率、双视图共识区域覆盖。
- `views.*.regions`：无正区域、全正/全负/全unknown；`overlap_conflict_fraction`为同视图正负窗口重叠比例。
- `inference_seconds`/`model_load_seconds`/`elapsed_seconds`：裁窗处理编码/加载/总耗时，不能混用。
- `internvl_reference`：只读旧结果并列检查；粒度和分数含义不同，不是精度横比。
- `effective_supervision_coverage=null`：可训练伪监督未接入、未测，区域覆盖不能替代它。

任一视图全unknown、任一图无共识、缺重复前景/明确背景、任一视图全正/全负、
或共同有效区域的变换冲突>0.25时需检查，以退出码2结束；unknown面积不能稀释冲突门禁。
真实负图可以全负，门禁不证明语义错误。全unknown一致率可为1，但共同有效一致率和正证据IoU
在空集时返回null，不能据此通过。退出0也只允许继续检查区域，不启动训练，不证明CLIP更准确。
不得为了非零覆盖降门槛或把未检出当负标签。

## 依据与本地验证边界

[CLIP论文](https://arxiv.org/abs/2103.00020)、[官方代码](https://github.com/openai/CLIP)、
[HF接口](https://huggingface.co/docs/transformers/model_doc/clip)支持图文对齐余弦检索；
本滑窗三态协议是项目诊断设计，原论文未验证它在本数据上的效果。
ProxyCLIP/SegEarth-OV的空间定位仍是后续候选，本次不叠加。
MM-DINO的冻结DINO与多层空间解码可供后续学生参考，但其像素监督不解决本次无标签教师矛盾。

此前本地仅用合成RGB/分数、模拟教师、随机初始化微型CLIP验证实现接口，
当时未加载真实CLIP权重。始终未连接云端、未读取真实人工标签，无真实语义质量或精度结论。

2026-10-18本地回归：102项全部通过（无跳过），包括翻转边缘/奇数坐标、显式背景与低分unknown、
重叠冲突、双视图支持、空集IoU、unknown不稀释冲突、单类退化阻断、CLI来源不符拒绝与失败保留。
随机微型CLIP在PyTorch2.8.0 CPU / Transformers4.57.6上验证了冻结、文本ensemble、批量图像编码接口；
没有用随机权重证明语义效果。编译、CLI帮助和差异检查通过。

2026-10-18离线准备补充：已在本地下载上述固定版本，权重字节数和SHA-256匹配官方LFS元数据，
7个配套文件的Git blob ID也匹配官方元数据。生成的离线压缩包为407726879字节（约408MB），
仅含8个模型文件、来源记录和校验清单；重新读取归档内容验证了各文件哈希，未包含下载缓存。
本地PyTorch2.8.0+cpu / Transformers4.57.6已实际加载该真实权重，在强制离线模式下，
对程序生成的512×512 RGB梯度及其翻转分别完成9窗编码，输出分数有限，模型参数保持冻结。
这只验证权重和项目编码接口可用，不验证真实光伏语义、云端GPU或用户截图中的Transformers5.16.1环境。

# 魔搭云服务器运行（soft_semantic v3）

当前方案仍是冻结 DINOv3、冻结 InternVL，只训练 Adapter 和 Decoder。
v4 修正粗网格到像素监督的转换、训练原型来源和缓存一致性，保留主干模型。

从仓库根目录执行下面命令。准备 CUDA 版 PyTorch；已有可用的云镜像时先检查
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`，
再安装其余依赖和项目：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

编辑 `configs/modelscope_soft_semantic.yaml` 的路径。它继承相同算法配置，
无需复制整份超参数；`extends` 相对当前 YAML 解析，嵌套字段合并、列表整体替换。
`exclude_manifest_file` 和 `exclude_rgb_hash_file` 相对声明它们的 YAML 解析，
其余相对路径相对运行命令时的工作目录。

运行使用新增的 `configs/modelscope_soft_semantic_v4.yaml`。它继承上述文件中你已经
配置的数据目录和分类名称，只把输出和 RGB staging 切换至 v4 新目录。
无需覆盖服务器上的 `/mnt/workspace/pv_data`、`困难负样本` 等本地设置。
v3 教师/伪标签不能直接当成 v4 训练缓存。

| 字段 | 默认位置 | 内容 |
|---|---|---|
| data.input_root | ./data | 能找到 pv-gf2 数据集的目录，分类目录下有 images；训练只读 RGB |
| data.exclude_manifest_file | ./data/history/split_manifest.json | 原历史教师缓存的 split_manifest.json，仅消费 RGB 文件名 |
| data.exclude_rgb_hash_file | 仓库内已有清单，继承基础配置 | 排除上一协议已使用的 120 张 RGB |
| data.rgb_staging_dir | ./runs/rgb_soft_v3 | 选出的 RGB 副本 |
| runtime.output_dir | ./runs/fresh120_soft_semantic_v3 | 特征、伪标签、模型、预测和报告 |
| runtime.model_cache_dir | ./model_cache | 模型缓存 |

历史 manifest 必须复制到服务器或改为实际路径；缺失会在 prepare 阶段报错。
这样保留原实验的样本隔离，避免迁移平台后重复使用已评估的图像。

将已准备的模型设置为本地路径，以下路径需要替换：

```bash
export DINOV3_MODEL_PATH=/path/to/dinov3/model.safetensors
export INTERNVL_MODEL_PATH=/path/to/InternVL3_5-4B
```

DINO 可使用 timm SAT493M 的 safetensors 文件（推荐旁边保留原 config.json），
也可指向完整的 Transformers 模型目录。InternVL 指向完整目录，包括 tokenizer、
配置、权重和模型所需的自定义 Python 文件。现有加载器直接读取这些本地文件；
未设置环境变量时仍使用配置中的 Hugging Face 模型 ID，不会自动切换下载源。

先用已有小测试的训练 RGB 检查教师（最多两张图、四次 global 推理）：

```bash
python scripts/11_teacher_preflight.py \
  --config configs/modelscope_soft_semantic_v4.yaml \
  --manifest runs/soft_smoke_002/rgb_manifest.json \
  --output-root runs/teacher_preflight_v4_001
```

该命令不提取 DINO、不训练、不读取人工标签。如果已有 manifest 在其他目录，
替换 `--manifest`；若尚未 prepare，先执行下方完整流程的 prepare 阶段，再传入
`runs/fresh120_soft_semantic_v4/rgb_manifest.json`。旧 RGB 可复用，旧教师响应不混用。
输出目录必须新建或为空；失败后保留，下一次换新目录。
查看 `preflight_report.json`、`reports/teacher_pair_checks.json` 和 `semantic_prompt.json`。
原图/翻转明确冲突超过 4/16 格时立即停止，四个 local 视图不会继续消耗时间。
通过只说明格式和变换一致性合格，不代表语义正确，更不保证已有双类训练种子。

然后运行 4 图完整测试：

```bash
python scripts/10_smoke_soft_semantic.py \
  --config configs/modelscope_soft_semantic_v4.yaml \
  --output-root ./runs/soft_smoke_v4_001 --samples 4
```

测试保留完整模型、每图六个教师视图和紧凑提示词；缩减为 4 图、A/B 两组、
单种子、2 个 epoch。它要求每个视图 16 格、每组至少 2 次成功优化，并验证封存。
查看 `runs/soft_smoke_v4_001/smoke_report.json`。不读取人工标签、不输出准确率。
v4 先运行教师再提取 DINO。伪标签正种子必须同时满足教师区域、RGB 和 DINO；
背景也需要明确多视图负证据。无合格正种子或背景种子时仍会正确停止，
而不会降阈值或制造标签。`prototype_seed_support.json` 区分原型缺种子与后续过滤失败。
测试目录必须新建或为空；测试失败后保留原目录，下一次换新目录。

通过后运行完整实验（120 图、4 组、3 种子、8 epoch）：

```bash
for stage in prepare teacher features pseudo train seal; do
  python scripts/08_label_free_pipeline.py \
    --config configs/modelscope_soft_semantic_v4.yaml --stage "$stage" || break
done
```

教师或伪标签报不完整网格时，先查看保存的 `semantic_prompt.json` 原始响应。
更改提示词、模型或训练协议后使用新的 output_dir 和 staging_dir；旧实验数据不覆盖。
普通运行不接受不完整缓存。仅检查旧截断恢复时，可给小测试添加
`--replay-cache /path/to/semantic_prompt.json`；此模式不加载模型或图片，也不验证 GPU 推理。

完整预测封存后，才单独启动最终评估：

```bash
python scripts/09_final_evaluation.py --output-root ./runs/fresh120_soft_semantic_v4
```

该进程先校验文件哈希和报告结构，再读取人工标签。小测试不自动执行这一步。

本次本地验证（2026-10-17）：78 项单元/回归测试通过，包含 6 张合成图像上
四组训练、完整教师缓存校验、预测封存和最终评估，及 AMP 溢出跳步检查。
此前回放了 4 张历史云端图像的 24 个响应，恢复 284 个完整单元；
24 个历史视图均不足 16 格，因此只能作为恢复测试，不能据此确认新版真实推理完整。
本次未在魔搭上加载真实权重或运行 GPU，未读取真实人工标签，没有新的准确率指标。

```bash
python -m unittest discover -s tests -p "test_*.py"
```

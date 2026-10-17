# 光伏遥感分割论文调研与当前方案建议

本轮落地更新：已根据用户回传的两图InternVL协议矛盾，新增单个冻结CLIP滑窗区域诊断，
保持DINO学生，不叠加SAM，不宣称复现ProxyCLIP或已获得像素伪标签。
固定窗口/文本/余弦margin，先比较变换、全空全满、区域证据覆盖和耗时；
最终可训练监督覆盖仍未接入。证据与运行命令见 [小实验记录](CLIP_WINDOW_PROBE_20261017.md)。
2026-10-18用户已完成真实CLIP两图推理，发现重复光伏窗口证据，也发现不同窗口重叠可造成区域一致性误读。
据此新增同坐标窗口统计及128/256固定尺度对照，保持同一CLIP；尚未生成像素伪标签或得到精度结果。
以下候选方法与论文结构建议不代表本轮已接入。

检索日期：2026-10-17。问题范围：目标数据训练过程不读取人工类别或像素标签，使用预训练模型生成监督，训练光伏分割模型。

## 对当前项目的判断

冻结 DINOv3 SAT、训练轻量 Adapter / Decoder 的主体结构有参考依据，可以保留。
优先需要替换或验证的是监督来源：InternVL 的 4×4 网格只表达区域内目标存在性，
无法直接提供像素占用率或边界；文本生成的 0.95 也没有经过概率校准。
云端已经出现原图全部 negative、水平翻转全部 positive，这足以说明该教师协议尚不稳定。

v4 修复了监督转换、原型污染、缓存来源和失败放行等实现问题，增加早停检查。
这些修复是有效实验的前提，不能证明教师会识别光伏。
本次论文调研没有再次启动实际模型推理，也没有产生新的真实数据准确率。

## 新增文献核对：MM-DINO 能借鉴什么

MM-DINO [14] 对本项目最有价值的是学生网络结构证据，而不是伪标签教师证据。论文冻结
DINOv3-ViT-L/16，从四个中间 Transformer block 提取序列特征，将其恢复为空间特征后用
1x1 投影和上/下采样构成多尺度金字塔；随后以 FRM 聚合多感受野上下文、以 FRN 做自顶向下和
迭代细化。多模态版本还使用可学习的早期跨模态权重和逐通道 ACFM。其消融中，冻结主干后的
Adapter、跨尺度增强与逐级细化均有贡献，因而支持“保留冻结 DINOv3、改进轻量空间头”的方向。

它不能直接解决当前无标签阻断。论文使用 Vaihingen、Potsdam、WHU-OPT-SAR 的像素真值，
把影像裁成 512x512，以交叉熵加 Dice 训练 50 epoch；文中的“zero-shot”是先在 Vaihingen
有监督训练、再不微调地测试 Potsdam，并非目标训练阶段无人工标签。LoRA 版本还会调整主干注意力，
也不同于本项目完全冻结 DINO 的当前协议。论文没有生成光伏语义或伪标签的模块，其类别和数据集
也不是光伏二类分割，因此不能用其 mIoU 证明本项目伪标签正确。

对照当前代码，`FrozenDINOv3.forward` 只返回最后一组 64x64 patch 特征，Decoder 是单尺度
卷积后放大到至多 256x256，再插值到 1024x1024；它没有 MM-DINO 的四层空间金字塔。
这是教师可用以后值得做的单独结构消融，但不应在没有正伪标签时先加入：更强 Decoder 无法从
零监督中恢复语义，而且会把教师变化与学生结构变化混在一起。论文的效率数据来自 48 GB L20，
不能直接外推到当前约 23 GB 的 A10。

本轮据此保持训练主体不变，先增加同一 InternVL、同一 RGB 预处理下的整图 presence 与 4x4
网格配对探针。若 presence 翻转稳定而网格反转，证据指向网格表达；若二者都反转，则应停止
InternVL 网格路线并优先换空间教师。presence 只作诊断门禁，不转成像素标签。
可选复用已有 DINO 特征，用当前融合和伪标签函数测量同一小批训练 RGB 的约束损失与最终覆盖；
缺双类或严重冲突即阻断，不松绑融合、不制造正类。94 项本地测试验证了实现，尚未在真实权重或数据上
证明该诊断方案能改善教师输出或分割质量。

## 与本任务最相关的路线

| 工作 | 如何产生空间监督 | 监督条件与接入限制 | 对本项目的价值 |
|---|---|---|---|
| A CLIP-Based Zero-Shot Photovoltaic Segmentation Framework for Remote Sensing Imagery，Remote Sensing，2026 [1] | 冻结 CLIP；LARA 聚合中间层注意力；CMSA 用图文梯度归因生成激活；CARS 用置信加权和尺度相关阈值生成掩膜 | 无下游权重训练；仍有提示词和超参数。未在读取的正文中找到明确的官方代码链接 | 直接针对光伏，比通用聊天式网格更贴近任务；适合做新教师的参考基线 |
| PANEL，ISPRS JPRS，2026 [2] | 在超过百万光伏图文对上预训练领域 CLIP；PANEL Surgery 输出光伏定位先验；KAM 将先验注入分割模型 | 零样本分支与 few-shot KAM 必须区分。后者使用少量标签；模型文件访问需接受作者条件 | 可优先测试其冻结的零样本语义定位，替换 InternVL 语义图；与当前语义 Adapter 的设计很接近 |
| ProxyCLIP，2024 [3]；CorrCLIP，2024/2025 [4] | CLIP 提供类别语义；DINO 等自监督特征提供空间对应；CorrCLIP 再以 SAM 区域约束 patch 交互 | 训练自由推理；论文的通用场景结果不等于光伏结果。原版使用的视觉特征不能未经验证直接换成 DINOv3 SAT | 为“语义识别”和“空间细化”分工提供明确依据，避免用 RGB 暗蓝规则定义光伏 |
| SegEarth-OV，2024 预印本 [5] | 面向遥感的 CLIP 密集推理，SimFeatUp 恢复空间细节，并减轻 patch 的全局偏置 | 目标任务无需额外训练；仍需注意领域、尺度及提示词依赖 | 可作为遥感专用的零训练基线，与通用 CLIP 的结果比较 |
| SAM 3，2025 [6]；SegEarth-OV3，2025/2026 [7] | SAM 3 接受概念文字或视觉提示，直接返回实例掩膜；SegEarth-OV3 融合实例/语义头，并用 presence 分数抑制不存在的类别 | 已有代码。SAM 3 有独立环境要求；SegEarth-OV3 论文明确按数据集调整部分阈值，不能据此声称完全不涉及任务配置选择 | 已有区域输出的教师候选，可绕开 JSON 网格生成；不保证能正确分割本数据的微小光伏 |

这里的“零样本”“零训练”主要指目标任务不再用人工标签训练。
CLIP、PANEL、SAM 等预训练模型本身使用过图文对或人工监督，不能表述为整个系统从未使用人工知识。

## 论文中哪些方法不能直接照搬

1. **WeCLIP / CLIP-ES 使用图像级类别标签。**
   WeCLIP [8] 使用冻结 CLIP 和可训练解码器，并动态细化伪标签，支持保留轻量解码器的选择。
   CLIP-ES [9] 采用 CAM、类别感知亲和传播和高置信区域损失，支持先定位、再细化、再训练的流程。
   两者的标准设定都有图像级标签；读取本数据的“正样本/负样本”目录当类别监督会改变当前协议。

2. **直接研究光伏的弱监督论文也可能需要真实标注。**
   EPUDA [10] 的摘要明确使用源域密集标注和目标域稀疏点标注，再通过边界适配和不确定性筛选进行迁移。
   可借鉴边界建模和冲突过滤，不能把其结果当成零人工目标监督的证据。
   此项已核对 DOI 和索引摘要，出版社接口仅返回元数据，未取得全文方法细节。

3. **“光伏潜力”不一定是“已有光伏板分割”。**
   LPL [11] 用图像级标注和 SAM 细化来识别潜在光伏位置，服务于分类后分割的候选区域筛选。
   SolarSAM 等屋顶潜力评估同样不能直接替代已有光伏组件的像素分割实验。

4. **DINOv3.seg 也不是无监督训练的现成答案。**
   2026 年的 dinov3.seg [12] 将 DINO 图文对齐、早晚两阶段细化和局部/全局推理结合起来。
   正文说明在 COCO-Stuff 118K 训练集上训练，使用密集分割损失；参考其结构需要说明外部监督来源。
   不能把“开放词汇”理解为“训练时不需要分割标签”。

5. **新模型也有遥感域偏移。**
   2026 年 SAM 3 遥感评测预印本 [13] 报告文字与视觉提示可能发生干扰，俯视语义和分辨率仍限制表现。
   这是近期实验报告，不能据此推断所有数据集或所有配置都会失败，但足以否定“换成 SAM 3 必然解决”的承诺。

## 推荐的下一版实现

优先做“有空间输出的教师 + DINO 空间约束”的小规模比较，保留当前训练主体。
这是一项新实验方案，尚未接入现有 v4 代码。

### 第一阶段：先验证教师，不训练学生

- 固定一小组训练 RGB 的内容哈希，例如 8–12 张；不按人工正负类别选图，不读 masks。
- 候选 A：冻结 PANEL 的零样本定位先验，或按 [1] / SegEarth-OV 的做法生成 CLIP 光伏激活。
  同时设置水体、暗色屋顶、阴影、温室等竞争背景描述，避免二分类只给一个“solar panel”词就将整图归前景。
- 候选 B：SAM 3 / SegEarth-OV3 的文字概念分割。固定少量光伏同义描述，保留实例掩膜、分数和 presence 输出。
- 每条路线都先检查原图与翻转对齐后的掩膜差异、有效前景覆盖、整图空/满、输出有限性及耗时。
  这些只衡量稳定性和可用性，不能作为无标签准确率。
- 提示词、阈值、权重版本及预处理进入来源记录；不使用冻结测试集的人工标签选择配置。
  如果稳定性门槛失败，停止该候选，不能用强制保留正样本来让训练启动。

### 第二阶段：区域细化和稀疏伪监督

1. 语义模型先给出光伏候选和明确背景，再用 DINOv3 SAT 的空间对应/相似度约束候选传播。
2. 如使用 SAM 1/2，它负责依据候选点或框细化边界，并不自行判断“这是光伏”。
   SAM 3 已具备概念识别接口，仍需验证其识别和掩膜是否适用于本数据。
3. 正种子来自语义支持与空间区域的一致部分，负种子来自明确背景；冲突、未覆盖和低证据区域保留 ignore。
   不将“没检测到”直接等同于“全图背景”。
4. RGB 规则只适合作为辅助先验，应评估其对非典型组件的漏检影响；不再将暗蓝色硬编码成光伏定义。
5. 注意特征空间兼容性：原始 DINO 特征与 CLIP/PANEL 文本向量不能直接做有语义意义的余弦相似度。
   ProxyCLIP 使用视觉特征的空间对应关系去组织 CLIP 特征；另一种可行接口是将两者都转换成对齐的空间图。
6. 同模型的多个视图或同源模型的相互同意都可能共享错误，一致性只能作为筛选信号，不能当成校准概率。

### 第三阶段：再训练和比较 Adapter

- 先用固定伪标签训练 DINO + Decoder 基线，再在同一伪标签、同一 split 上比较语义 Adapter。
- 保留容量对照、打乱语义图对照，避免把参数量或额外教师信息的收益误算成 Adapter 结构收益。
- 当前 semantic Adapter 在测试时仍使用测试 RGB 的教师语义图，应如实记录教师成本。
  如果目标是让部署只运行 DINO + 学生头，需要另做去教师蒸馏实验。
- 有限高质量种子通常比冲突区域的强制填充更适合启动训练；但覆盖过低时应报告监督不足，不能宣称训练有效。
- 所有配置锁定并封存预测后，再执行独立人工标签评估。不同论文的数据分辨率、标注粒度和划分不同，不能直接横比 IoU。

实际接入顺序建议：先试可取得权重的领域语义教师；若 PANEL 文件权限暂不可用，
可从 CLIP / ProxyCLIP 思路的小样本基线开始。SAM 3 / SegEarth-OV3 作为另一条独立候选路线，
先测试后决定，不需要同时把所有大模型常驻在 A10 显存中。

## 来源与核验范围

以下是论文、作者仓库、出版社正文或作者机构页面。方法建议是对这些工作的综合分析，并非已经完成的复现。

1. **Li et al., 2026. A CLIP-Based Zero-Shot Photovoltaic Segmentation Framework for Remote Sensing Imagery.**
   [DOI / 论文](https://doi.org/10.3390/rs18060865)；
   [出版社完整 XML](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00865/article_deploy/remotesensing-18-00865.xml)。
   已读取方法、实验设置、参数表及讨论。论文报告多分辨率光伏实验，但也进行参数敏感性分析；
   不把其数据集上的最优参数搜索过程迁移到我们的冻结测试标签上。
2. **Deng et al., 2026. PANEL: A photovoltaic-specific vision-language model for zero-shot and few-shot PV interpretation tasks in remote sensing.**
   [DOI](https://doi.org/10.1016/j.isprsjprs.2026.05.012)；
   [作者机构摘要](https://ira.lib.polyu.edu.hk/handle/10397/119607)；
   [作者模型卡和文件目录](https://huggingface.co/UEXdo/PANEL)。
   已核对摘要、模型卡和约 599 MB 的权重文件条目，未下载权重。模型卡可读，文件访问需要接受条件；未取得全文。
3. **Lan et al., 2024. ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation.**
   [论文](https://arxiv.org/abs/2408.04883)；[全文 HTML](https://arxiv.org/html/2408.04883)。
   已核对训练自由设定、空间对应模块和滑窗推理。
4. **CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation.**
   2024 年预印本，2025 年修订。
   [论文](https://arxiv.org/abs/2411.10086)；[作者代码](https://github.com/zdk258/CorrCLIP)。
   已核对 CLIP、SAM 区域约束和自监督特征的职责。
5. **Li et al. SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images.**
   2024 年预印本。
   [论文](https://arxiv.org/abs/2410.01768)。已核对摘要；早期项目站点当前返回 404，未将失效站点作为可运行代码证据。
6. **Carion et al., 2025. SAM 3: Segment Anything with Concepts.**
   [论文](https://arxiv.org/abs/2511.16719)；[官方代码和环境要求](https://github.com/facebookresearch/sam3)。
   已核对概念分割能力和官方示例。官方仓库当前要求 Python 3.12+、PyTorch 2.7+、CUDA 12.6+，
   应单独建立环境并先确认云端兼容性。
7. **Li et al. SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images.**
   2025 年预印本，2026-04 修订。
   [论文](https://arxiv.org/abs/2512.08730)；[作者代码](https://github.com/earth-insights/SegEarth-OV-3)。
   已读取方法、实现设置和仓库说明。论文明确对各数据集略调背景阈值与初始置信阈值；
   主实验覆盖通用遥感任务，不能据此认定已验证我们的光伏数据。
8. **Zhang et al., CVPR 2024. Frozen CLIP: A Strong Backbone for Weakly Supervised Semantic Segmentation (WeCLIP).**
   [论文](https://arxiv.org/abs/2406.11189)；[作者代码](https://github.com/zbf1991/WeCLIP)。
   已核对摘要及 CVPR 官方论文条目；标准设定使用图像级标签。
9. **Lin et al., CVPR 2023. CLIP Is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation (CLIP-ES).**
   [论文](https://arxiv.org/abs/2212.09506)；[作者代码](https://github.com/linyq2117/CLIP-ES)。
   已核对摘要及 CVPR 官方论文条目；标准设定使用图像级标签。
10. **Edge-preserving and uncertainty-guided domain adaptation for weakly supervised rooftop photovoltaic mapping (EPUDA), ISPRS JPRS, 2026.**
    [DOI](https://doi.org/10.1016/j.isprsjprs.2026.07.008)。已核对 DOI 元数据和索引摘要，未取得全文。
11. **Yang et al., 2024. A novel weakly-supervised method based on the segment anything model for seamless transition from classification to segmentation: A case study in segmenting latent photovoltaic locations.**
    [DOI](https://doi.org/10.1016/j.jag.2024.103929)；[作者代码](https://github.com/Github-YRQ/LPL)。
    已核对索引摘要及仓库对应关系；未取得出版社全文。
12. **dinov3.seg: Open-Vocabulary Semantic Segmentation with DINOv3, 2026 预印本.**
    [论文与全文](https://arxiv.org/html/2603.19531)。已读取架构、损失与实验训练配置。
13. **Promptable Concept Segmentation from Above: Evaluating SAM 3's Zero-Shot and One-Shot Capabilities in Remote Sensing, 2026 预印本.**
    [论文](https://arxiv.org/abs/2607.09583)。已核对摘要中的遥感域限制与提示模态干扰结论；未独立复现。
14. **Qin et al., 2026. MM-DINO: DINOv3-Based Universal Framework for Uni and Multimodal Remote Sensing Image Semantic Segmentation.**
    IEEE Transactions on Geoscience and Remote Sensing, vol. 64, article 4407012，
    [DOI](https://doi.org/10.1109/TGRS.2026.3677346)。已核对完整正文、Fig. 2--4、训练设置、
    消融和跨数据集实验；论文声明代码将发布于 [GitHub](https://github.com/KimotaQY/MM-DINO)。

本次仅访问公开论文、模型卡和代码元数据，没有访问 Kaggle、没有下载大模型、没有读取真实人工标签。

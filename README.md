# Monkey Ethology Agent

笼内猕猴行为分析流水线：**DeepLabCut 姿态追踪 → 规则行为分类 → 学术可视化 → 行为标注视频**。

本仓库提供可配置的分析 Agent。追踪标记点按实验指定；行为标签为基于阈值的 locomotion ethogram（Static / Walking / Climbing），并可选刻板与转圈检测。参数通过 YAML 与 CLI 暴露。个体数 / 天数不固定，由 `study` 配置或 CLI 决定。

---

## 目录

- [功能概览](#功能概览)
- [处理流程](#处理流程)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置](#配置)
- [DeepLabCut 集成](#deeplabcut-集成)
- [行为分析](#行为分析)
- [参数说明](#参数说明)
- [命令行](#命令行)
- [输出说明](#输出说明)
- [可视化](#可视化)
- [视频标注](#视频标注)
- [Python API](#python-api)
- [目录结构](#目录结构)
- [常见问题](#常见问题)

---

## 功能概览

1. 按用户指定的 bodyparts 创建或驱动 DeepLabCut 项目。
2. 读取 DLC `.h5` / `.csv`，或已扁平化的关键点表。
3. 关键点预处理：遮挡插值、异常修复、时序平滑。
4. 滑窗提取运动学特征（自适应重心）。
5. 将行为划分为 **Static**、**Walking**、**Climbing**。
6. 可选：刻板（stereotypy）、转圈（circling）。
7. 导出 ethogram、时间预算、轨迹、热图、标准对比图与汇总表。
8. 将分类结果叠加到原始视频（骨架 + `Behavior: …` 标签）。

入口：`EthologyAgent`（`src/monkey_ethology/agent.py`）与 CLI `python -m monkey_ethology`。

---

## 处理流程

```text
视频（可选）
    │
    ▼
DeepLabCut  ──►  h5 / csv
    │
    ▼
ingest  ──►  扁平化为 Joint_x / Joint_y / Joint_p
    │
    ▼
preprocess  ──►  遮挡 → 异常 → 中值平滑
    │
    ▼
extract_features  ──►  逐窗 COM 运动学特征
    │
    ├──► locomotion（Static / Walking / Climbing）
    ├──► stereotypy（可选）
    └──► circling（可选）
    │
    ├──► visualize       ──►  图件 + CSV 汇总
    └──► annotate-video  ──►  原始视频叠加骨架与行为标签
```

已有关键点表时，可直接运行：

```bash
python -m monkey_ethology run \
  --input path/to/keypoints.csv \
  --animal M01 \
  --cage default \
  --output ./output/run_M01
```

`--animal` 可以是任意个体编号（仅写入结果元数据）；未提供同名阈值文件时，一律使用通用阈值。

---

## 环境要求

| 组件 | 说明 |
|------|------|
| Python | ≥ 3.9 |
| 核心依赖 | `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `h5py`, `opencv-python-headless` |
| DeepLabCut | 可选；仅 `track` 训练/推理需要 |
| OpenCV | 视频标注（`annotate-video`）需要 |
| Streamlit | 可选；用于 `apps/viz_studio.py` |

---

## 安装

```bash
cd monkey-ethology-agent
python -m pip install -e .

# 可选扩展
python -m pip install -e ".[dlc]"   # DeepLabCut
python -m pip install -e ".[viz]"   # Streamlit
python -m pip install -e ".[dev]"   # pytest
```

安装检查：

```bash
python -m monkey_ethology init-config --dest ./my_config.yaml
python -m pytest tests/test_core.py -q
python examples/run_synthetic.py
```

---

## 快速开始

### 姿态追踪（DeepLabCut）

标记点需显式指定。未安装 DeepLabCut 时，`--dry-run` 会生成占位 `config.yaml` 与 API 调用日志。

```bash
python -m monkey_ethology track \
  --videos /path/to/cage.mp4 \
  --bodyparts "Head,Back,Tail(Root)" \
  --steps create_project,set_bodyparts,extract_frames \
  --dry-run \
  --output ./output/tracking
```

使用扩展骨架模板：

```bash
python -m monkey_ethology track \
  --videos /path/to/cage.mp4 \
  --bodyparts-file configs/bodyparts/primate_extended.yaml \
  --steps create_project,set_bodyparts \
  --dry-run \
  --output ./output/tracking
```

DLC 推理完成后转为扁平表：

```python
from monkey_ethology import EthologyAgent, load_config

agent = EthologyAgent(load_config())
agent.convert_dlc(["/path/to/DLC_resnet50_....h5"])
```

### 行为分类与出图

```bash
python -m monkey_ethology run \
  --input /path/to/DLC_or_flat.csv \
  --animal M01 \
  --cage default \
  --plots ethogram,trajectory,heatmap,publication_figure \
  --language zh \
  --formats png,pdf \
  --output ./output/run_M01
```

批量处理目录：

```bash
python -m monkey_ethology run \
  --input /path/to/folder \
  --pattern "*.csv" \
  --animals M01,M02 \
  --days 1,2,3 \
  --output ./output/batch
```

个体数与天数不固定，在 `configs/default.yaml` 的 `study` 段配置（或用上面的 CLI 覆盖）：

```yaml
study:
  animals: ["M01", "M02"]   # 空 [] = 自动发现全部个体
  days: [1, 2, 3]           # 空 [] = 自动发现全部天数
  filename_template: "{animal}-day{day}"
```

### 仅可视化

```bash
python -m monkey_ethology visualize \
  --segments path/to/id-day1_features-segment.csv \
  --keypoints path/to/id-day1_processed.csv \
  --features path/to/id-day1_features.csv \
  --plots ethogram,time_budget,bout_duration,transition,com_trajectory_heatmap,body_center \
  --language en \
  --output ./output/figures
```

### 行为标注视频

视频文件需与会话 stem 同名（如 `562-day1.MP4`），目录由 `--video-dir` 或 `video_annotation.video_dir` 指定。

```bash
# 一键流水线中叠加标注（分类完成后自动写 annotated_videos/）
python -m monkey_ethology run \
  --input /path/to/folder \
  --pattern "*.csv" \
  --animals 562,883 \
  --days 1,2,3 \
  --video-dir /path/to/videos \
  --output ./output/batch

# 已有 sessions 时单独标注
python -m monkey_ethology annotate-video \
  --video-dir /path/to/videos \
  --data-dir ./output/batch/sessions \
  --stem 562-day1 \
  --output ./output/batch

# 预览前 N 帧
python -m monkey_ethology annotate-video \
  --video-dir /path/to/videos \
  --data-dir ./output/batch/sessions \
  --stem 562-day1 \
  --max-frames 750 \
  --output ./output/batch
```

叠加样式：关节点 + 骨架颜色随行为变化（Walking=绿，Climbing=蓝，Static=黑），帧上显示 `Behavior: {label}`；低置信度点标红。

### 交互式调图

```bash
streamlit run apps/viz_studio.py
```

---

## 配置

默认值见 `configs/default.yaml`。每次运行会在输出目录写入 `resolved_config.yaml`（合并后的生效配置）。

### 合并顺序

1. `configs/default.yaml`
2. `--config`
3. `--bodyparts-file` / `--cage` / `--animal` / `--viz-config`
4. CLI 参数（`--bodyparts`、`--plots`、`--language` 等）
5. Python：`cfg.set("dotted.key", value)`

### 内置模板

| 路径 | 用途 |
|------|------|
| `configs/bodyparts/primate_minimal.yaml` | Head, Back, Tail(Root) |
| `configs/bodyparts/primate_extended.yaml` | 扩展灵长类骨架 |
| `configs/cages/default.yaml` | 当前唯一笼子几何（25 fps，720×406） |
| `configs/animals/default.yaml` | 通用 locomotion 阈值（任意个体共用） |
| `configs/visualization.yaml` | 可视化默认项 |
| `configs/demo_study.yaml` | 演示用 study + 视频标注叠加配置 |

`--animal` 是个体标签，不必为每只动物建文件。仅当某只需要单独调参时，再复制 `animals/default.yaml` 为 `{id}.yaml`。

### 全局 CLI 选项

适用于 `track`、`run`、`classify`、`visualize`、`annotate-video`：

| 参数 | 说明 |
|------|------|
| `--config` | 主 YAML |
| `--bodyparts-file` | 标记点 YAML |
| `--bodyparts` | 逗号分隔点名 |
| `--cage` | `default` 或路径；省略则用主配置 `cage` 段 |
| `--animal` | 任意个体 ID（标签）或阈值 YAML 路径 |
| `--animals` | 逗号分隔个体列表，覆盖 `study.animals` |
| `--days` | 逗号分隔天数，覆盖 `study.days` |
| `--video-dir` | 原始视频目录（与 stem 同名） |
| `--no-annotate` | 跳过行为标注视频 |
| `--viz-config` | 可视化 YAML |
| `--output` | 输出根目录（同时作为 DLC 工作目录） |
| `--plots` | 逗号分隔图类型 |
| `--language` | `zh` \| `en` |
| `--formats` | 如 `png,pdf` |
| `--dpi` | 出图 DPI |

---

## DeepLabCut 集成

实现：`src/monkey_ethology/dlc/project.py`（`DLCProjectManager`）。

各步骤为 DeepLabCut 2.x/3.x 官方 API 的薄封装。未安装 DLC，或使用 `--dry-run` / `dlc.dry_run: true` 时，调用记录写入 `dlc_api_log.yaml`，并生成占位项目配置。

### 步骤与 API 对应

| `--steps` 名称 | DeepLabCut API | 作用 |
|----------------|----------------|------|
| `create_project` | `deeplabcut.create_new_project` | 创建项目与初始 config |
| `set_bodyparts` | 编辑 `config.yaml` | 写入 bodyparts / skeleton / pcutoff |
| `extract_frames` | `deeplabcut.extract_frames` | 抽帧供标注 |
| `label_frames` | `deeplabcut.label_frames` | 标注 GUI（需显示器） |
| `check_labels` | `deeplabcut.check_labels` | 标注检查 |
| `create_training_dataset` | `deeplabcut.create_training_dataset` | 生成训练集 |
| `train_network` | `deeplabcut.train_network` | 训练 |
| `evaluate_network` | `deeplabcut.evaluate_network` | 评估 |
| `analyze_videos` | `deeplabcut.analyze_videos` | 推理 → h5/csv |
| `filterpredictions` | `deeplabcut.filterpredictions` | 时序滤波 |
| `create_labeled_video` | `deeplabcut.create_labeled_video` | 叠加预览视频 |

另提供精炼相关封装：`extract_outlier_frames`、`refine_labels`、`merge_datasets`。

### 传入 DLC 的配置项

**项目**（`project.*`）：`name`、`experimenter`、`working_directory`、`copy_videos`、`videotype`。

**抽帧**（`dlc.*`）：`extract_mode`（`automatic` \| `manual`）、`extract_algo`（`kmeans` \| `uniform`）、`userfeedback`、`numframes2pick`。

**训练**（`dlc.*`）：`net_type`、`augmenter_type`、`shuffle`、`displayiters`、`saveiters`、`maxiters`、`pcutoff`。

**推理**：`save_as_csv`、`videotype`、输出目录位于 `--output` 下。

已安装 DeepLabCut 时的示例：

```bash
python -m monkey_ethology track \
  --videos v1.mp4,v2.mp4 \
  --bodyparts "Head,Back,Tail(Root)" \
  --steps create_project,set_bodyparts,extract_frames,label_frames,check_labels,create_training_dataset,train_network,evaluate_network,analyze_videos,filterpredictions,create_labeled_video \
  --output ./output/dlc_project
```

---

## 行为分析

分类为**阈值规则 + 信号特征**，非 SVM / 随机森林 / HMM 训练管线。实现位于 `preprocess/`、`features/`、`classify/`。

### 坐标系

图像原点在左上角：**x 向右增大，y 向下增大**。较小的 `mean_com_y` 对应画面中偏上位置。攀爬判定使用 `cage.climbing_y`（小笼模板默认 `202`）。

### 预处理

| 模块 | 方法 |
|------|------|
| 遮挡 | 检测 `p < p_threshold` 的连续段；线性插值，并施加位移 / 速度 / 边界约束 |
| 异常 | 越界裁剪；大跳变用邻帧平均或限速修复 |
| 平滑 | 低置信帧保持上一高置信坐标，再对 x/y 做一维中值滤波 |

### 特征提取（`AdaptiveCOMExtractor`）

以 `features.window_size` 帧为不重叠窗口（默认 **13** @ 25 fps ≈ 0.52 s）。

- 窗内平均 `p ≥ p_threshold` 的关节为可用点。
- COM = 可用点 `(x, y)` 均值。
- 朝向取 `orientation_pairs` 中第一对可用点。
- 每窗汇总：速度、位移、COM 统计、头尾 Δy、角速度、FFT 周期性、身体面积等。

`*_features.csv` 中 `window_index` 为特征窗编号。视频帧 ≈ `window_index × window_size`。segment 的 `start` / `end` 同为窗索引。

### Locomotion（Static / Walking / Climbing）

两阶段分段（`classify/locomotion.py`）。

**阶段 1 — 静止 vs 运动**

- 满足（`mean_velocity > velocity_threshold` **且** `displacement > displacement_threshold`），或相对 `window_threshold` 窗前的持续垂直 COM 变化且速度足够 → 计为运动。
- 连续运动窗 ≥ `window_threshold` → `state = 1`，否则 `state = 0`。

**阶段 2 — 行走 vs 攀爬**（运动段内，子窗长度 `subwindow_size`）

1. 不满足运动条件 → `pattern = 0`（`potential_static`）。
2. `mean_com_y < climbing_y` → Climbing（`pattern = 3`）。
3. Δx 小且 Δy 大，或头尾 Δy 过大 → Climbing（`pattern = 3`）。
4. 否则 → Walking（`pattern = 2`）。

CSV 中 pattern：`0` Static，`2` Walking，`3` Climbing。`reasons` 列为规则命中说明。

### 刻板（可选）

约 3 s 窗、1 s 步长的四特征：周期性（FFT）、角速度标准差、圆拟合误差、累积角位移。默认：高周期性 **且** 高角速度波动 **且**（圆误差小 **或** \|累积角\| > 2π）。关闭：`stereotypy.enabled: false`。

### 转圈（可选）

对选定部位（默认 Head）滑窗：ROI 内有效点比例、半径范围、半径变异系数、累积转角 ≥ `min_angle`。关闭：`circling.enabled: false`。

---

## 参数说明

默认见 `configs/default.yaml`。个体覆盖见 `configs/animals/{id}.yaml`。

### `bodyparts`

| 键 | 含义 |
|----|------|
| `names` | 追踪点列表 |
| `skeleton` | DLC / 叠加用骨架边 |
| `com_joints` | 参与 COM 的点 |
| `orientation_pairs` | 朝向优先点对 |
| `head_joint` / `tail_joint` | 头尾差与刻板轴向 |
| `sitting_joints` | 坐姿遮挡启发式 |
| `aliases` | 列名别名（如 Tail → Tail(Root)） |

### `cage`

| 键 | 小笼默认 | 含义 |
|----|----------|------|
| `fps` | 25 | 帧率 |
| `width` / `height` | 720 / 406 | 分辨率 |
| `climbing_y` | 202 | COM y 小于此值判攀爬 |
| `pixel_to_cm` | 0.3202 | 像素 → 厘米 |
| `session_max_sec` | 1800 | 分析时长上限（`0` 表示不截断） |

### `preprocess` / `features`

| 键 | 默认 | 含义 |
|----|------|------|
| `p_threshold` | 0.4 | 低置信阈值 |
| `min_occlusion_duration` | 3 | 最短遮挡帧数 |
| `max_displacement` / `max_velocity` | 100 / 50 | 跳变 / 速度上限 |
| `smooth_window` | 5 | 中值窗；`≤0` 关闭 |
| `features.window_size` | 13 | 特征窗长度 |

### `locomotion`

| 键 | 默认（通用） | 含义 |
|----|------------------|------|
| `velocity_threshold` | 10.0 | 速度门限 |
| `displacement_threshold` | 4.0 | 位移门限 |
| `window_threshold` | 5 | 连续运动窗数 |
| `head_tail_difference_threshold` | 75.0 | 头尾差攀爬 |
| `com_x_threshold` / `com_y_threshold` | 5.5 / 10.5 | 垂直运动攀爬 |
| `subwindow_size` | 4 | 细分子窗 |

### `stereotypy` / `circling` / `visualization`

完整列表见 `configs/default.yaml`（含 `periodicity_threshold`、`circle_error_threshold`、转圈 ROI、`plots`、`colors` 等）。

---

## 命令行

```text
python -m monkey_ethology {track|run|classify|visualize|annotate-video|init-config} [options]
```

| 子命令 | 作用 |
|--------|------|
| `track` | DeepLabCut 步骤执行（`--videos`、`--steps`、`--dry-run`） |
| `run` | ingest → preprocess → features → classify → visualize →（可选）视频标注 |
| `classify` | 同 `run`，支持 `--no-viz` |
| `visualize` | 由已有 segment / keypoint / feature 出图 |
| `annotate-video` | 用 processed + features-segment 在原始视频上叠加行为 |
| `init-config` | 导出默认可编辑 YAML |

---

## 输出说明

```text
output/<run>/
  resolved_config.yaml
  sessions/<stem>/
    <stem>_processed.csv
    <stem>_features.csv
    <stem>_features-segment.csv
    <stem>_stereotypy.csv          # 若启用
    <stem>_circling.csv            # 若启用
  figures/
    <stem>_ethogram.png|pdf
    <stem>_publication.png|pdf
    <stem>_behavior_summary.csv
    ...
  dlc_api_log.yaml                 # track 后生成
```

segment 时间（秒）：

\[
t \approx \frac{\mathrm{start} \times \mathrm{window\_size}}{\mathrm{fps}}
\]

---

## 可视化

| `plots` 名称 | 内容 |
|--------------|------|
| `ethogram` | 行为时间谱 + 饼图 |
| `raster` | 多会话 ethogram 栅格 |
| `time_budget` | 各态时间占比 |
| `bout_duration` | 片段时长分布 |
| `transition` | 转移矩阵 |
| `trajectory` | 按行为着色的 COM 轨迹 |
| `heatmap` | 驻留热图 |
| `velocity` | 速度时序 |
| `features` | 特征时序 |
| `position` | 笼内分区占比 |
| `quality` | 置信度曲线 |
| `stereotypy` | 四特征刻板面板 |
| `circling` | 转圈事件叠加 |
| `publication_figure` | 综合拼图 |
| `com_trajectory_heatmap` | COM 轨迹 + 驻留热图（标准图） |
| `body_center` | 身体中心轨迹面板 |
| `position_analysis` | 上下/左右分区路程与速度 |
| `behavior_comparison` | 多会话行为状态对比 |

批量 `run` 目录时，会额外写出 cohort 级图：`all_sessions_behavior_comparison`、`all_sessions_position_analysis`、按个体的 raster。

---

## 视频标注

实现：`src/monkey_ethology/viz/video_annotate.py`（`BehaviorVideoAnnotator`）。

将 `*_processed.csv` 关节点与 `*_features-segment.csv` 行为段叠加到原始视频：

- 骨架颜色：Walking 绿、Climbing 蓝、Static 黑
- 帧中心附近显示 `Behavior: {label}`
- likelihood < `video_annotation.p_threshold` 的点标红
- segment 的 `start`/`end` 为窗索引，乘 `features.window_size` 映射到帧

主要配置（`configs/default.yaml` → `video_annotation`）：

| 字段 | 说明 |
|------|------|
| `enabled` | 是否在 `run` 中自动标注 |
| `video_dir` | 原始视频目录（文件名=`{stem}.mp4` / `.MP4` 等） |
| `output_subdir` | 默认 `annotated_videos` |
| `draw_labels` | 是否绘制关节点名称与置信度 |
| `show_animal_on_head` | Head 旁显示动物 ID |
| `max_frames` / `start_frame` | 预览裁剪；`null` 表示整段 |

演示脚本：`scripts/run_demo_annotate_video.sh`。

---

## Python API

```python
from monkey_ethology import EthologyAgent, load_config

cfg = load_config(cage="default", animal="M01")
cfg.set_bodyparts(["Head", "Back", "Tail(Root)"])
cfg.set("visualization.plots", ["ethogram", "heatmap", "publication_figure"])
cfg.set("io.output_dir", "./output/my_run")

agent = EthologyAgent(cfg)
# agent.track(["video.mp4"], steps=["create_project", "set_bodyparts"])
result = agent.run("M01-day1.csv", animal_id="M01", video_dir="./videos")
# result["annotated_video"]  # 若找到同名视频
```

分步调用：

```python
kp = agent.ingest("M01-day1.csv")
kp = agent.preprocess(kp)
feat = agent.extract_features(kp)
seg = agent.classify_locomotion(feat, animal_id="M01")
agent.visualize("M01-day1", seg, keypoints=kp, features=feat)
agent.annotate_video("M01-day1", kp, seg, video_dir="./videos", animal_id="M01")
```

---

## 目录结构

```text
monkey-ethology-agent/
├── configs/                 # YAML 默认与模板（含 demo_study.yaml）
├── scripts/                 # 演示：出图 / 视频标注
├── assets/                  # 可视化背景图等
├── src/monkey_ethology/
│   ├── agent.py             # EthologyAgent
│   ├── cli.py
│   ├── config.py
│   ├── study.py             # 个体/天数筛选
│   ├── dlc/                 # DeepLabCut 封装
│   ├── io/
│   ├── preprocess/
│   ├── features/
│   ├── classify/
│   └── viz/                 # 学术图 + video_annotate
├── examples/
├── apps/viz_studio.py
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## 常见问题

**未安装 DeepLabCut**  
`run` / `classify` / `visualize` 可直接处理已有表格。`track --dry-run` 用于生成项目配置与 API 日志。

**图中中文缺字**  
将 `visualization.font_family` 设为系统可用的中文字体，或使用 `--language en`。

**Walking 很少**  
查看 `*_features.csv` 中 `mean_velocity` / `displacement`，按个体与笼子调整 `locomotion.*`。

**segment 索引与帧的关系**  
`start` / `end` 为特征窗索引；乘以 `window_size` 得到帧号。

**找不到标注视频**  
确认 `--video-dir` 下存在与 stem 同名的视频（如 `562-day1.MP4`），且已生成 `*_processed.csv` 与 `*_features-segment.csv`。

**新个体 / 新笼子 / 不定天数**  
在 `study.animals` / `study.days` 中列出本次要跑的个体与天数（数量任意）；留空则自动发现。  
新个体阈值：直接共用通用 locomotion 默认值，或复制 `configs/animals/default.yaml` 为 `{id}.yaml`。  
换笼子：改 `configs/cages/default.yaml`（或主配置 `cage` 段）。

---

## License

对外发布前请补充许可证与引用信息。

# 源码阅读指南

公子，源码已经按功能职责拆分。先读命令入口，再按下面的数据流阅读；像素和频域方法共享目标函数、代理模型和优化调度，避免修改一个基线时另一种方法发生无意偏移。

```text
frequency_rag/                  项目主包
├── cli.py                      命令入口及参数分派
├── image_selection/            选图与测试题准备
│   ├── data.py                 冻结图像清单及旧选图数据读取
│   ├── dataset.py              原始多会话数据、问答与图片审计
│   ├── scoring.py              原有目标图评分与词汇门控
│   ├── categories.py           四类注入测试的语义约束
│   ├── candidates.py           候选池筛选、视觉验证和问题生成
│   └── plans.py                注入／投毒计划及扰动清单导出
├── pixel_attack/               原像素对抗扰动
│   └── update.py               梯度符号更新、像素范围及扰动预算投影
├── frequency_attack/           频域对抗扰动
│   └── dct.py                  余弦基底、系数更新、频带投影和渐进扩频
├── attack_core/                两类攻击共享的核心
│   ├── engine.py               方法选择、优化循环、检查点和图片保存
│   ├── objective.py            全局／局部／文本损失与目标特征缓存
│   ├── ot.py                   局部聚类及最优传输匹配
│   └── surrogates.py           攻击代理编码器、权重加载及可微特征
├── memory_agent/               记忆智能体
│   ├── llm.py                  大语言模型服务、视觉描述和结构化输出
│   ├── encoder.py              记忆用的图文联合编码器
│   ├── agent.py                逐轮记忆写入、检索和携带证据回答
│   ├── scenarios.py            干净／目标图／对抗图等条件构造
│   └── backends/               可动态选择的五种后端
│       ├── base.py             统一接口、记忆条目及检索结果结构
│       ├── registry.py         后端名称注册、惰性加载和扩展入口
│       ├── vector.py           公共向量存储、图遍历与持久化
│       ├── murag.py            平面多模态检索
│       ├── ngmemory.py         相似度图记忆
│       ├── augustus.py         概念标签图记忆
│       ├── universalrag.py     模态路由记忆
│       └── mem0.py             外部事实记忆服务适配
├── evaluation/                 评估与结果统计
│   ├── vectors.py              图像／问题向量及相似度指标
│   ├── judge.py                与受害智能体分离的问答裁判
│   ├── memory_metrics.py       问答、检索及条件攻击指标
│   └── results.py              跨方法对比、预算核验及结果文件
├── experiments/               实验执行流程
│   ├── attack_pipeline.py      扰动生成、向量评估和旧选图核查
│   ├── memory_cli.py           记忆实验子命令
│   └── memory_runner.py        多条件记忆实验、逐题执行和产物管理
└── common/                     公共基础设施
    ├── config.py               攻击配置及参数约束
    ├── io.py                   文件与图片读写、校验和摘要
    ├── numerics.py             公共向量运算
    ├── paths.py                包目录与项目根目录
    ├── profiling.py            时间和显存测量
    └── provenance.py           时间戳及递归源码快照
```

目录树中的英文是实际文件和目录标识，同行给出中文职责。单元测试仍位于项目根目录的 `tests`（测试目录），运行生成的数据位于 `outputs`（运行产物目录）；二者不混入可安装源码。

## 四类问题是怎样生成的

执行 `memory-select-pairs`（筛选注入图像对命令）后，程序先使用配置的编码器，对外部候选池的图片和描述进行排序。然后读取类别约束，把源图和目标图一起发给视觉语言模型，要求它判断图像是否符合类别，并输出：上传语句、问题、真实答案、目标答案和目标图描述。

调用链：

```text
cli.py
  → experiments/memory_cli.py：dispatch_memory
  → image_selection/candidates.py：select_injection_pairs
  → memory_agent/llm.py：ChatModel.complete
  → 配置的模型服务
  → 验证结构化输出并写入图像对文件
```

`ChatModel`（聊天模型适配器）通过聊天补全兼容协议发送文字及图片，已接入 MiniMax（稀宇科技大模型服务）中国版配置。不是随机拼接问题，也不是仅靠文字描述验证图片。它需要候选池、真实图片、可用模型配置与密钥环境变量；缺少条件时无法生成正式题集。

`--per-category 1`（每类一组）会要求生成四类各一组，共四道问题；默认值为四，即四类共十六道。数量不足时明确失败，不保证不合格的候选池也能凑满数量。

以下命令只是使用示例，本次重构没有执行远端模型请求；路径指向您后续准备的真实文件。

```powershell
python tools/run_cli.py memory-select-pairs --pool data/pool.local.json --memory-config configs/memory.local.json --per-category 1 --output outputs/four_category_pairs.json
```

该命令只产生经过视觉验证的图像对与问题。随后用 `memory-plan`（生成实验计划命令）插入指定领域的会话并导出扰动清单，再运行扰动和记忆实验。原问题与目标答案不进入受害智能体的隐藏上下文；目标答案仅供独立裁判判分。

## 五种后端怎样动态接入

调用者统一使用 `make_memory`（记忆后端工厂），配置的 `backend`（后端名称）决定加载哪个实现。注册表只在使用时导入选中的模块，事实记忆依赖不会在选择平面后端时初始化。

| 后端配置值 | 对应文件 | 关键差异 |
| --- | --- | --- |
| `murag`（平面多模态记忆） | `backends/murag.py` | 原始图文向量检索 |
| `ngmemory`（神经图记忆） | `backends/ngmemory.py` | 相似度建边和图遍历 |
| `augustus`（概念图记忆） | `backends/augustus.py` | 模型提取概念，结合概念覆盖排序 |
| `universalrag`（模态路由记忆） | `backends/universalrag.py` | 模型选择无检索、文本或图像通道 |
| `mem0`（事实记忆服务） | `backends/mem0.py` | 视觉描述后由外部服务提取、保存及检索事实 |

扩展第六种后端时，实现统一的写入、检索、重置、保存、恢复接口，再用 `register_backend`（注册记忆后端）提供构造工厂；工厂接收配置、编码器和语言模型。不能静默覆盖已注册的后端。

## 迁移兼容范围

命令名称、命令参数、配置中的攻击方法和后端名称保持不变。包顶层导出的攻击函数和配置接口也保持不变。直接导入旧内部模块路径的外部脚本需要按下表修改；不在根目录保留一排空壳转发文件，以保持职责清晰。

| 旧模块文件 | 当前位置 |
| --- | --- |
| `attacks.py` | `attack_core/engine.py`，像素更新抽到 `pixel_attack/update.py` |
| `frequency.py` | `frequency_attack/dct.py` |
| `data.py`、`selection.py` | `image_selection/data.py`、`image_selection/scoring.py` |
| `models.py`、`objective.py`、`ot.py` | `attack_core/` 下对应文件，模型文件改名为 `surrogates.py` |
| `memory_preparation.py` | `image_selection/candidates.py`、`image_selection/plans.py` |
| `memory_dataset.py` | `image_selection/` 下类别、数据、计划模块，以及 `memory_agent/scenarios.py` |
| `memory_models.py` | `memory_agent/llm.py`、`memory_agent/encoder.py` |
| `memory_backends.py` | `memory_agent/backends/` 下统一接口、五种实现和注册表 |
| `memory_experiment.py` | `experiments/memory_runner.py`、`memory_agent/agent.py`、`evaluation/` 下裁判与指标 |
| `evaluation.py`、`benchmark.py` | `evaluation/vectors.py`、`evaluation/results.py` |
| `pipeline.py`、`memory_cli.py` | `experiments/attack_pipeline.py`、`experiments/memory_cli.py` |
| `config.py`、`io.py`、`profiling.py` | `common/` 下同名文件 |

源码追溯已改为递归扫描，记录相对于包或项目根目录的路径，不会遗漏嵌套后端，也不会因为多个目录都有同名初始化文件而覆盖摘要。旧实验快照属于历史记录；本次重构产生新的源码标识，旧、新代码结果不应跳过一致性检查直接混合比较。

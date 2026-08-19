# 交互组件方案（Interactive Widgets Plan）

## 目标

为教程中"看公式看不会、动手滑两下就懂"的概念配交互演示，挂载在文档页右侧面板，不打断正文阅读。

## 布局与交互模型

- **触发**：正文中概念出现处嵌入一行标记 `<div class="widget-mount" data-widget="组件id" data-title="显示名"></div>`（goldmark `unsafe=true` 原样通过），渲染成一个小按钮卡"▶ 交互演示：xxx"。
- **面板**：点击后右侧 dock 出现面板（桌面端 `position:fixed` 右侧 420px、可折叠、多组件切换 tab；<1180px 时变成底部抽屉，与现有 docs-toc 的断点一致）。面板 sticky 跟随阅读。
- **无 JS 降级**：按钮不出现，正文完全不受影响。

## 技术选型

- 纯 vanilla JS + Canvas/SVG，**无框架、无构建步骤**（站点是纯静态 Hugo，现状无打包器，保持一致）。
- 文件布局：
  - `site/static/widgets/panel.js` — 面板框架 + 注册表 + 按需加载
  - `site/static/widgets/widgets.css` — 纸面 brutalist 风格（ink 描边、米白底、蓝色强调，与站点一致）
  - `site/static/widgets/w-<id>.js` — 每个组件一个文件，暴露 `mount(container)`，由注册表动态注入
- 预计算数据（token 向量、CIFAR 缩略图等）由 `scripts/prepare_widget_data.py` 生成到 `site/static/assets/widgets/`，产物入库，CI 不需要 torch。
- 质量门：`tests/test_docs.py` 增加 `data-widget` 引用与注册表一致性检查；组件文件语法用 `node --check` 在本地验证。

## 组件清单（分期交付）

### P0 框架 + 三个试点（先验证管线）
| id | 位置 | 交互 |
|---|---|---|
| `linear-fit` | B1 任务一 | a/b 滑杆，canvas 重画散点+直线；实时 MSE 数值、∂L/∂a ∂L/∂b 梯度箭头、损失等高线小图 |
| `conv-sweep` | B2 task_11 | 左输入图、上方 3×3 kernel 可拖动、右特征图逐格补满；联动 im2col 矩阵视图 |
| `kv-cache` | B3 task_30 | prefill/decode 步进动画：每层 cache 增长、RoPE 位置推进、新旧 K/V 高亮 |

### P1 六个
| id | 交互 |
|---|---|
| `mlp-boundary` (B1) | 编辑 2-h-2 权重，决策边界与隐藏层二维投影实时更新 |
| `backprop-steps` (B1) | 2-4-4-2 网络逐步动画：前向点亮激活 → 反向梯度流 → 权重更新量 |
| `optimizer-race` (B1) | SGD/Momentum/Adagrad/RMSProp/Adam 在同一等高线面上赛跑；lr、β 可调 |
| `norm-playground` (B1) | BatchNorm vs LayerNorm：拖动 batch 样本，看归一化前后分布与 running stats |
| `image-batch` (B2) | 8 张 CIFAR 斜排书架陈列 → 点击展开 RGB 三通道 → 点像素显示 NCHW 布局值 |
| `causal-mask` (B3) | 训练并行（全 mask 一步）vs 推理逐 token 的 mask 演进动画 |

### P2 七个
| id | 交互 |
|---|---|
| `pool-sweep` (B2) | MaxPool/AvgPool 窗口扫描交互，梯度路由高亮 |
| `im2col-grad` (B2) | im2col 矩阵 + col2im 梯度累加演示 |
| `token-embed-3d` (B3) | 预计算 token 向量投影到 3D 坐标系，可旋转 |
| `qkv-flow` (B3) | X→Q/K/V→scores→softmax→加权求和 分步 shape 动画 |
| `pos-encoding` (B3) | 输入一句话 → sinusoidal PE 热图 + 三角函数坐标系；RoPE 旋转版对照 |
| `gqa-map` (B3) | 4 query heads → 2 组 KV 的映射与缓存动画 |
| `sampling` (B3) | temperature / top-k / top-p 对分布形状的实时影响（自加） |

## 实施方式

1. 先落 P0：建框架三件套 + 三个试点组件 + 对应文档标记 + 测试，整体一次提交验证端到端。
2. P1/P2 每个组件一个子智能体实现（组件文件互不冲突；文档标记由主线统一插入），逐批提交。
3. 每批完成后：本地 `hugo` 构建 + headless Chrome 截图逐组件验收 + 全测试套件。

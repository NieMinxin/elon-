# Polymarket 自动买卖与 Elon 帖子区间系统

项目包含两个独立入口，均使用 Polymarket 官方统一 Python SDK
`polymarket-client`，并默认只进行纸面交易：

- `polymarket-bot`：基于固定价格阈值的单市场机器人。
- `elon-arb`：针对 Elon Musk 七日发帖数量区间盘口的概率交易系统。

`elon-arb` 不是无风险套利。它用历史数据估计区间概率，再与包含手续费的盘口价格比较，
因此仍然承担模型误差、统计口径变化、数据延迟、滑点和流动性风险。

## Elon 区间系统

系统以本地 MySQL 中的 xTracker 帖子作为唯一训练集，不抓取 X 网页。周模型把历史
拆成与目标盘口同长度的滚动窗口，并学习星期/小时基线。运行期间模型参数保持固定，
每两分钟用当前累计数量以及最近 1/3/6/12/24 小时和 1/3/7/14/30 天的数量，重新加权
历史相似轨迹。这是实时概率更新，不是在线重新训练，可以同时响应短时激增、沉寂和
均值回归。

系统的目标不是承诺命中某一个区间，而是尽量把最终帖子数落在概率最高的三个连续
20帖区间内，并在持仓期间判断原三档是否仍值得持有。当前采用的完整数据流是：

```text
xTracker帖子 -> MySQL -> 每周拟合历史模型 -> 最终帖子数概率分布
                    \-> 每2分钟实时特征 -----^          |
Polymarket订单簿 -> 可成交价格/深度/费用 ---------------+-> 持有、预警或滚仓评估
```

周模型与实时监控各自负责不同的问题：

- 周模型只使用预测时点之前的历史数据，提供正常状态下的星期、小时和七日总量分布。
- 两分钟监控不重新训练模型，而是依据本轮已累计帖子数、近期速率、激增程度和沉寂
  时间，对历史相似轨迹重新加权。
- 最终输出是每一个20帖区间的概率，而不只是一个最终数量点估计。
- 从所有可交易区间中，选择概率和最大的三个连续区间作为当前目标组合。
- 目标组合变化只产生观察/报警信号；在持续性、迟滞和成本门槛完整实现前，不自动
  滚仓。

### 实现状态与安全边界

已经实现并验证的部分：

- xTracker 历史回填、按数据库最大日期增量拉取，以及按帖子 ID 幂等去重。
- MySQL 数据质量审计、每周重新拟合、盘中自适应预测和无未来数据泄漏的走步回测。
- 每两分钟读取 xTracker 累计数量以及 Polymarket 15个区间的批量订单簿。
- 保存可成交 Bid/Ask 和两侧各10档深度，用于估算真实卖出价值、滑点和滚仓成本。
- 三个连续区间的选择，以及滚动一档、两档或三档时需要卖出和买入哪些区间的评估。
- 区间变化、发帖状态变化、回撤和亏损阈值的本地报警记录；配置密钥后可发钉钉。
- 最佳三档变化时，按同一数据截点运行路由基线、原GRU-NLL、V3密集小时GRU和V4
  校准集成，并把四组均值、80%区间、各档概率与三档合计概率放进同一条钉钉提醒。

目前安全运行方式是 `--observe-only`。以下环节仍需完成并经过纸面回归测试：

- 目标区间持续多轮后才确认滚仓的状态机，以及防止相邻组合来回跳动的迟滞机制。
- 同时处理三档持仓的完整纸面建仓、仓位导入和一至三档滚仓执行。
- 使用累计的历史订单簿数据，回测手续费、滑点、换手率、组合收益和最大回撤。
- 上述结果达标后才考虑真实下单；当前不要为 Elon 三档策略开启 `--live`。

### 回测结果如何理解

当前报告使用最近90天构造了84个按天向前移动、长度均为7天的历史模拟盘口。每一个
模拟盘口都只使用当时已经可见的数据；新模型每7天重新拟合一次。由于窗口每天移动
一次，相邻样本会重叠，因此84个结果不是84个完全独立事件。
本次报告的训练与验证数据截至 `2026-10-13 16:00 UTC`，共使用13,291条盘前帖子；
完整机器可读结果保存在 `elon-model-report.json`。

| 已观察时间 | 旧模型三档覆盖率 | 新模型三档覆盖率 | 旧模型 MAE | 新模型 MAE |
| ---: | ---: | ---: | ---: | ---: |
| 24小时 | 47.6% | 63.1% | 54.6 | 36.1 |
| 42小时 | 50.0% | 61.9% | 47.6 | 34.2 |
| 84小时 | 59.5% | 73.8% | 37.8 | 29.3 |
| 126小时 | 79.8% | 88.1% | 19.5 | 16.9 |

“已观察24小时”表示一个七日盘口开始24小时后，用当时已经发生的帖子预测第168小时
的最终总数；42、84和126小时同理。“三档覆盖率”表示模型选择的三个连续20帖区间
中，至少有一个包含最终结果。例如目标三档是 `320-339 / 340-359 / 360-379`，最终
结果为352就算覆盖，结果为381则不算。

“MAE 36.1”表示最终数量点预测与真实结果平均相差约36.1帖，方向可能高估也可能低估。
它不是区间宽度，也不是盈利百分比。旧模型是在回测期开始前一次拟合的静态经验窗口
模型；新模型会每周只用届时可见的历史重新拟合，并结合盘口已经观察到的近期发帖状态
修正剩余时间预测。

这些数字的正确结论是：观察时间越长，剩余不确定性通常越小，新模型选择的三档更
容易覆盖最终结果。它们不能解释为“有88.1%的概率盈利”。实际盈亏还取决于买入价、
卖出时的可成交 Bid、流动性、手续费、滑点以及是否过早滚仓。

### 为什么读取 Polymarket 盘口

帖子数据负责估计概率，Polymarket 数据负责回答“这个概率是否值得交易”。系统每两
分钟读取盘口不是为了训练发帖模型，而是为了取得：

- 每个区间当前可成交的买入和卖出价格，从而比较模型概率与市场隐含概率。
- 实际订单簿深度，从而估算 `$20` 买入或整仓卖出会产生多少滑点。
- 三档组合此刻能真正卖出的净值，而不是页面中间价或最后成交价。
- 卖掉旧档并买入新档后的手续费与滑点，以判断滚仓带来的模型价值提升能否覆盖成本。

因此，即使模型认为中心区间已经移动，也不应仅凭预测立即滚仓。系统还需要看到移动
持续存在，并确认按真实订单簿计算后的新组合改善足够大；否则继续持有通常更有利。

### 本地帖子数据库

`elon-data` 只使用 xTracker，不调用 X API。MySQL 中保存 xTracker 帖子 ID、X 帖子
ID、完整正文、发帖时间、xTracker 捕获时间、metrics 和原始 JSON。时间统一为 UTC，
帖子表使用 X 帖子 ID 作为主键，重复同步会更新已有记录而不会插入副本。

先在当前 PowerShell 会话设置本机 MySQL 连接信息：

```powershell
$env:MYSQL_HOST = "127.0.0.1"
$env:MYSQL_PORT = "3306"
$env:MYSQL_DATABASE = "polymarket_bot"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
```

初始化数据库并回填过去一年：

```powershell
.\.venv\Scripts\elon-data.exe init-db
.\.venv\Scripts\elon-data.exe backfill --days 365
```

以后执行增量同步。程序读取数据库中最大的 `created_at`，从该时间点开始（包含等于）
拉取到当前时间；`platform_post_id` 是主键，因此相同帖子只会更新，不会重复插入：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\scripts\sync_xtracker_to_mysql.ps1
.\.venv\Scripts\elon-data.exe status
```

也可以手工指定包含式开始日期：

```powershell
.\scripts\sync_xtracker_to_mysql.ps1 -StartDate "2026-10-17T00:00:00Z"
```

如果要重新检查延迟入库的数据，可以主动向前重叠 48 小时；帖子 ID 去重仍然生效：

```powershell
.\scripts\sync_xtracker_to_mysql.ps1 -OverlapHours 48
```

建议用 Windows 任务计划程序每周执行一次 `elon-data.exe sync`。活跃盘口期间，交易
程序每两分钟增量读取 xTracker、按帖子 ID 写回 MySQL，并使用 xTracker tracking total
作为官方累计数量。

训练前先检查数据库质量：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv\Scripts\elon-model.exe audit
```

执行按时间顺序的 90 日走步回测。回测分别模拟观察七日盘口的 24、42、84 和 126
小时，并报告 MAE、RMSE、80% 预测区间覆盖率、单档命中率和连续三档覆盖率：

```powershell
.\.venv\Scripts\elon-model.exe backtest
```

报告保存为 `elon-model-report.json`。查看当前事件预测、实时状态和区间概率：

```powershell
.\.venv\Scripts\elon-model.exe forecast
```

### 五阶段模型实验

`elon-model experiment` 是独立于线上监控的研究入口。它使用真实的26个盘口结算边界、
多个时间 fold、七日 block bootstrap 和逐样本走步预测，同时对照简单的近期24小时
速率外推基线。实验不会修改线上模型，也不会产生订单：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv\Scripts\elon-model.exe experiment `
  --folds 3 `
  --fold-days 60 `
  --bootstrap-samples 300
```

每次运行都会创建新的 `experiments/<UTC运行时间>/`，不会覆盖旧实验：

```text
manifest.json                 实验时间范围、fold、随机种子和盘口边界哈希
predictions.jsonl             每个模型、时点和样本的完整26档概率
weights.jsonl                 每次拟合参数、邻居权重、GBM树桩和校准参数
summary.json                  汇总指标及相对近期速率基线的配对置信区间
checkpoints/<model>/          单模型中间结果，长实验可检查进度
stages/01-.../report.json     五个阶段各自的对照报告
```

如果只调整报告逻辑，可以直接从已保留的逐样本结果重新汇总，不重新训练：

```powershell
.\.venv\Scripts\elon-model.exe experiment-report `
  --run-dir experiments\20261017T065005.310846Z
```

五个阶段分别验证：

1. Brier、log loss、RPS、ECE、可靠性曲线，以及近期速率和静态经验基线。
2. 有效样本量、Beta后验下界和基于已结算历史样本的选择偏差修正。
3. 官方 tracking、posts 端点与 MySQL 对账；另用声明过的2%端点延迟压力测试比较
   修正前后模型，压力测试不冒充真实历史数据。
4. 三个60日 fold、每个观察时点162个样本、七日 block bootstrap、分层指标和逐样本
   日志；区间固定为 `<20` 至 `500+` 的真实结算边界。
5. 时序温度校准、负二项模型、纯 Python GBM 加时间顺序 conformal 残差分布。

首轮完整实验位于 `experiments/20261017T065005.310846Z`。下面每格依次为
`MAE / log loss / 三档实际命中率`：

| 已观察时间 | 近期速率基线 | 自适应模型 | GBM + conformal |
| ---: | ---: | ---: | ---: |
| 24小时 | 58.1 / 3.950 / 35.2% | 50.9 / 2.590 / 42.6% | 43.3 / 2.460 / 38.9% |
| 42小时 | 54.3 / 4.050 / 37.7% | 46.2 / 2.480 / 46.3% | 38.8 / 2.350 / 44.4% |
| 84小时 | 39.1 / 3.740 / 52.5% | 34.7 / 2.260 / 60.5% | 33.0 / 2.190 / 56.8% |
| 126小时 | 20.4 / 2.500 / 74.7% | 18.1 / 1.570 / 83.3% | 24.3 / 2.030 / 70.4% |

本轮结论不是“复杂模型必胜”：

- 近期速率基线的三档平均预测概率为64.6%至97.7%，实际命中只有35.2%至74.7%，
  明显过度自信；它适合作为必须战胜的基线，不适合直接定价。
- 自适应模型四个时点的三档平均预测概率为38.2%、42.8%、52.1%、78.2%，实际命中
  为42.6%、46.3%、60.5%、83.3%，总体略偏保守。
- 当前 Beta winner 修正和温度校准都使 ECE 变差，因此实验判定为不采用，不能接入
  入场规则。概率下界机制已保留，但需要重新设计先验和选择偏差估计。
- GBM 在24至84小时改善点预测和 log loss，但126小时明显退化。它只能作为早中期候选
  模型，不能替换全时段自适应模型。
- 负二项模型表现混合且校准较差，当前不升级为生产模型。

相对近期速率基线，自适应模型四个时点的 log loss 改善，其配对七日 block bootstrap
95%区间均大于0；MAE改善区间多数仍跨0。完整置信区间、分层结果和每次模型权重以
JSON保留，后续调整必须创建新实验运行，不能覆盖本轮证据。

### GBM / adaptive 路由策略实验

路由策略不是比较两个单模型的平均分，而是固定按观察时间选择预测器：24、42和84小时
使用 `gbm_conformal`，126小时使用 `adaptive_reconciled`。它直接复用同一批逐样本
checkpoint，不重新训练，也不修改源实验：

```powershell
.\.venv\Scripts\elon-model.exe experiment-route `
  --source-run experiments\20261017T065005.310846Z `
  --bootstrap-samples 300
```

首次组合回放位于 `experiments/routes/20261017T114424.058216Z`，共648个预测样本。
整体结果如下：

| 策略 | MAE | log loss | 三档命中率 | ECE |
| --- | ---: | ---: | ---: | ---: |
| GBM 24–84h + adaptive 126h | **33.30** | **2.141** | 55.86% | **0.026** |
| 全程 GBM | 34.85 | 2.256 | 52.62% | 0.032 |
| 全程 adaptive | 37.46 | 2.224 | **58.18%** | 0.055 |
| 近期速率基线 | 42.98 | 3.565 | 50.00% | 0.294 |

配对七日 block bootstrap 显示，路由策略相对全程 GBM 的 MAE 改善为 `1.55`
（95% CI `0.50–2.83`），log loss 改善为 `0.115`（`0.034–0.221`），三档命中率提高
`3.24`个百分点（`1.00–5.71`）。相对全程 adaptive，MAE和log loss也改善，但三档
命中率降低 `2.31`个百分点，95% CI为 `-6.03–1.23`，差异尚不确定。

这条路由规则是在查看源实验结果后提出，因此 manifest 明确标记
`post_hoc_exploratory=true`、`independent_validation=false`。当前结论是“值得进入下一轮
预注册回测”，不能作为独立样本外证据直接替换生产模型。

实时监控默认设置 `ELON_FORECAST_STRATEGY=route`，并按事件开始后的已观察时间选择模型：

| 已观察时间 | 实时预测模型 |
| ---: | --- |
| 0至不足24小时 | `gbm_conformal_0h` 冻结基线 + `adaptive` 实时似然比修正 |
| 24至不足42小时 | `gbm_conformal_24h` |
| 42至不足84小时 | `gbm_conformal_42h` |
| 84至不足126小时 | `gbm_conformal_84h` |
| 126小时及以后 | `adaptive_reconciled_126h` |

系统每两分钟同步 xTracker 当前累计数与帖子时间、读取全部盘口订单簿，用路由模型重算最终
帖子数分布，并选择概率和最高的三个连续区间。目标三档、发帖状态或路由模型发生变化时
会触发钉钉报警；存在纸面/导入持仓时还会计算可成交平仓价值、峰值回撤、持仓恶化以及
按当前 Bid/Ask 滚仓后的模型价值变化。路由策略目前被程序强制限制为 `--observe-only`，
即使误设 `ELON_ENABLE_LIVE=true` 也不会进入真实下单流程。

### 第三版实验：统一 PMF、公平协议和动态计数模型

第三版使用独立命令，不修改第二版结果，也不替换线上监控模型：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv\Scripts\elon-model.exe experiment-v3 `
  --folds 3 `
  --fold-days 60 `
  --bootstrap-samples 300 `
  --simulation-paths 1200
```

只修改汇总或校准口径时，可直接从保留的 checkpoint 重建报告，不连接 MySQL、也不重新
训练模型：

```powershell
.\.venv\Scripts\elon-model.exe experiment-v3-report `
  --run-dir experiments\v3\20261017T103258.157269Z
```

第三版的计算口径和实验约束如下：

- 每个预测先生成从当前累计数开始的整数 PMF；均值、标准差、档位概率和80%区间都从
  同一个 PMF 计算。事件结束后 PMF 严格收敛为当前累计数的单点分布。
- 三个连续60日区段依次标为 `tuning`、`validation`、`final_test`。候选模型每七天
  重拟合；旧经验模型同时保留静态拟合和每周拟合版本，避免把更新频率误当成模型收益。
- 每日重叠窗口作为主样本，同时单独报告每七日一个的非重叠窗口。final test 每个观察
  时点有54个每日样本、8个非重叠样本；后者样本很少，只作为稳健性检查。
- `ideal_complete_history` 假定历史帖子即时可见，是理想上界；`as_known` 只使用
  `imported_at <= observed_at` 的帖子。历史官方 tracking 快照不存在，因此不能还原
  当时的官方累计数，这一限制记录在 manifest。
- 相似轨迹实验分别比较累计窗口、互不重叠窗口、静默时间、速率加速度、连续活跃时长、
  爆发分数、特征权重、90日历史半衰期和邻居数量。历史样本不按 ISO 周分组：从每次
  拟合的历史终点开始，按 `4天、3天、4天、3天...` 向后移动终点，每个终点取此前
  7天作为一个相似轨迹窗口。有效样本量按这些七日窗口的实际重叠比例修正；新颖度高或
  重叠修正 ESS 不足时与稳健经验基线混合并降低信号置信度。归一化权重为 `w_i` 时，
  `ESS = 1 / sum_i sum_j(w_i * w_j * overlap(i,j) / 7天)`；完全不重叠时退化为普通
  加权 ESS，完全重合时 ESS 为1。
- 动态负二项模型按小时模拟活动状态的持续和回落，并允许方差大于均值。对数活动转回
  发帖率时包含 Jensen 校正，避免模拟噪声系统性抬高长期均值。
- validation 通过 log loss、Brier、三档命中率、80%覆盖误差、区间宽度、三档切换率
  和 ECE 的等权排名选择模型。相似轨迹与动态模型的混合权重也只用 validation 的
  样本外 log loss 学习；final test 不参与选择。

修正后的有效运行位于 `experiments/v3/20261017T103258.157269Z`。旧运行
`experiments/v3/20261017T084943.579439Z` 使用 ISO 周 ESS，只保留为对照，不能作为
当前模型。下面是新运行未参与选择的
`final_test` 每日样本，格式为 `MAE / log loss / 三档实际命中率`：

| 已观察时间 | 近期速率 | 每周经验模型 | 90日半衰期轨迹模型 | 动态负二项 |
| ---: | ---: | ---: | ---: | ---: |
| 24小时 | 57.7 / 4.268 / 46.3% | 51.0 / 2.552 / 53.7% | **38.6 / 2.447 / 57.4%** | 56.0 / **2.419** / 40.7% |
| 42小时 | 61.1 / 4.415 / 29.6% | 46.2 / 2.469 / 48.1% | **35.0 / 2.498 / 61.1%** | 51.8 / **2.384** / 37.0% |
| 84小时 | 46.3 / 4.508 / 46.3% | 37.6 / 2.268 / 61.1% | **28.1 / 2.115 / 77.8%** | 39.3 / **2.059** / 64.8% |
| 126小时 | 22.9 / 3.137 / 72.2% | 18.1 / 1.628 / 77.8% | **15.5 / 1.460 / 88.9%** | 20.0 / 1.495 / 87.0% |

validation 的多指标排名选择 `analog_half_life_90`。四个观察时点学到的轨迹模型权重
均为1.0，因此最终融合没有加入动态模型。动态模型在 final test 的84和126小时 log loss
更低，但这是测试后观察，不能据此回改权重；应在下一次预注册实验里验证随剩余时间变化
的融合规则。

`as_known` 回放显示数据延迟会实质性损害概率质量。90日半衰期轨迹模型在24、42、84、
126小时的 log loss 分别从理想口径的 `2.447/2.498/2.115/1.460` 变为
`2.470/2.842/2.306/2.046`。因此不能只引用完整历史口径的成绩来设置交易阈值。

本次有效运行保留18,360条预测和29组 checkpoint。整数 PMF 最大归一误差为
`2.22e-16`，26个真实盘口区间的最大归一误差同为 `2.22e-16`，所有报告的80%区间
实际概率质量都至少为80%。全部12,960条相似轨迹记录满足
`0 < overlap ESS <= 原始权重 ESS`，新记录不含 ISO 周 ESS 字段。
`experiments/v3/20261017T075246.531537Z` 是发现动态模型 Jensen 偏差的失败运行，
原样保留用于审计，不应作为模型结论。

### 开盘前基础模型（baseline-0h）

基础模型使用专门命令训练，只预测“事件刚开始、已观察帖子数为0”时的完整最终数量
分布，不再用24小时模型外推开盘时点：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv\Scripts\elon-model.exe experiment-baseline
```

候选包括每周经验分布、四个相似轨迹版本和 `gbm_conformal_0h`。所有候选使用相同的
每日滚动七日目标、每七天重拟合，以及 `tuning / validation / final_test` 三段顺序
切分。Analog 历史窗口按4天、3天交替向后推进，不按 ISO 周。模型选择只读取
validation，并按以下加权排名求和：log loss 3、Brier 2、三档 ECE 2、80%覆盖误差1、
三档命中率1、MAE 1、80%区间宽度0.5。final test 不参与选择。

首次正式运行位于 `experiments/baseline/20261017T143058.857094Z`。MySQL共有13,598条
记录；排除首个不完整日后实际训练使用13,543条，截止到 `2026-10-17 11:47:56 UTC`，
保留972条逐样本预测和789条模型/邻居权重记录。validation 选择
`gbm_conformal_0h`：

| 模型 | validation MAE | validation log loss | validation三档命中 | final-test MAE | final-test log loss | final-test三档命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gbm_conformal_0h` | **35.6** | **2.266** | **48.1%** | 46.8 | 2.670 | 48.1% |
| `analog_half_life_90` | 46.4 | 2.456 | 38.9% | 45.0 | 2.595 | **66.7%** |
| `analog_disjoint` | 49.2 | 2.469 | 42.6% | **44.6** | **2.566** | 61.1% |
| `empirical_weekly` | 90.3 | 2.792 | 25.9% | 72.3 | 2.654 | 50.0% |

相对 `empirical_weekly`，GBM 在 final test 的 MAE 改善25.6帖，七日 block bootstrap
95%区间为8.1至50.2；但 log loss 改善为-0.016，区间为-0.264至0.361，三档命中率
变化也不确定。因此当前胜者主要提高中心预测，尚不能证明概率定价全面更好。manifest
明确记录 `independent_validation=false`，仍需用未来纸面盘口验证。

运行时在事件开始时重新补齐盘前七天数据并冻结 `P0`。前24小时不重新训练，而是计算：

```text
Pt(k) ∝ P0(k) × [Adaptive_t(k) / Adaptive_0(k)] ^ strength
strength = sqrt(min(已观察小时 / 24, 1))
```

似然比限制在 `1/20` 至 `20`，防止极小尾部概率放大噪声。24小时后进入既有
GBM-24/42/84与Adaptive-126路由。这一“P0 + 实时似然比”组合是新接线，尚没有独立
历史回测，所以保持只读纸面监控；不得据此开启真实下单。

若目标盘口尚未开始，`production/baseline.json` 会标记
`provisional_before_event_start`。当前20–27日暂定输出为均值204.2、80%区间132–294、
三档160–179 / 180–199 / 200–219（41.8%）；它包含未知盘前时段，事件开始时必须由
程序自动刷新并冻结，不能直接作为入场信号。

### GRU剩余数量实验

`experiment-gru` 是独立的研究入口，不会修改实时路由或提交订单。它先将帖子时间清洗
为UTC小时序列，再用精确七日总数生成监督标签。训练窗口按4天、3天交替向后推进；
验证和最终测试只使用互不重叠的七日事件，并在训练、验证和测试边界保留七天隔离带。

早期V1/V2实验同时比较并完整保留：

- `route_frozen`：使用同一训练截止时间冻结的现有GBM/Adaptive路由。
- `mlp_nb_nll`：只使用累计数、剩余时间和近期速率等聚合特征。
- `dual_gru_nb_nll`：盘前28天与盘口内序列分别编码的小型双GRU。
- `dual_gru_nb_rps`：在上一模型基础上增加区间RPS顺序损失。

后续实验没有覆盖旧结果：V2增加非对称混合分量和日级盘前编码；V3把训练事件从五个
检查点扩展为每6小时一个检查点，但每个事件的总训练权重仍为1；V4训练五个不同随机
种子的密集小时级GRU，平均完整整数PMF，并且只使用各fold验证集选择温度参数。
当前命令默认执行V4，仍保留单模型作为同场对照。

神经模型预测剩余帖子数的负二项混合分布，推理时加上当前累计数，再通过现有
`CountForecast.probability(lower, upper)` 聚合到任意盘口区间。当前第一版使用
`hidden_size=16`，每个事件的多个观察时点合计权重为1，避免把两分钟或多检查点快照
误当成独立事件。

截至2026年10月18日，同一组3-fold、12个独立最终测试事件的结果为：

| 版本 | MAE | Log loss | Brier | 三档覆盖率 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 冻结路由基线 | **35.72** | 2.527 | **0.869** | **53.3%** | 继续作为生产基线 |
| V2日级GRU-NLL | 40.88 | 2.377 | 0.874 | 51.7% | 小幅结构改善 |
| V3密集小时GRU-NLL | 39.75 | **2.273** | 0.874 | 48.3% | 当前最佳GPU概率模型 |
| V4五成员校准集成 | 42.24 | 2.343 | 0.881 | 51.7% | 更稳但未超过V3 |

V3改善主要集中在42小时和84小时，但对应路由选择不被验证集支持；V4的bootstrap区间
仍跨过0。任何GPU版本都未同时超过路由基线的MAE、Brier和三档覆盖，因此全部保持
challenger状态，不接入两分钟生产监控。下一轮改进需要更多独立七日事件或新的外生
特征，不能继续用这12个最终测试事件挑结构或权重。

#### WhiteHouse泛化验证

使用[xTracker WhiteHouse](https://xtracker.polymarket.com/user/WhiteHouse)做了两种严格
区分的泛化测试：一是保持V4结构和超参数不变、仅在WhiteHouse历史上重新训练；二是
完全不重训，直接把Elon V4的权重、特征缩放器和Elon验证集选出的温度用于WhiteHouse。
后者才是严格的跨账号zero-shot权重迁移。

WhiteHouse在2026年1月16日至3月5日之间存在约1,153小时的xTracker覆盖缺口。清洗器
不会把这段缺失误当作零发帖，而是丢弃缺口之前的数据；本次实验实际使用
2026年3月6日至10月16日的6,260条帖子。评估仍采用3个时间fold、每fold 4个互不重叠
七日最终测试事件、7天embargo，以及0/24/42/84/126小时五个观察点，共60条测试预测。

| WhiteHouse测试 | MAE | Log loss | Brier | 三档覆盖率 | 80%区间覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WhiteHouse冻结路由基线 | **23.48** | **1.818** | 0.830 | 71.7% | 75.0% |
| WhiteHouse重训单GRU | 23.81 | 2.050 | **0.785** | 75.0% | 70.0% |
| WhiteHouse重训五成员集成 | 26.30 | 2.255 | 0.816 | **78.3%** | 75.0% |
| WhiteHouse重训校准集成 | 26.55 | 2.535 | 0.831 | **78.3%** | 71.7% |
| Elon单GRU zero-shot | 82.35 | 2.893 | 0.933 | 28.3% | 50.0% |
| Elon五成员集成 zero-shot | 73.54 | 2.679 | 0.912 | 38.3% | 61.7% |
| Elon校准集成 zero-shot | 70.82 | 2.783 | 0.918 | 38.3% | 58.3% |

重训结果说明双编码器结构在另一个账号上能够学习，但目前不稳定：未校准集成在fold 0/1
的Log loss为1.371/1.622，优于路由的1.723/1.738；fold 2却恶化到3.772，路由仅
1.994。相对路由的事件级bootstrap区间全部跨0，所以不能认定结构稳定胜出。

Elon权重的zero-shot迁移则明确失败。未校准集成在开盘时对WhiteHouse的平均预测为
328.3，实际均值为211.5，MAE达到121.8；到126小时有当前累计数作为锚点后，MAE才降到
29.1，仍差于路由的13.4。zero-shot集成相对路由的MAE、RPS和三档覆盖率bootstrap
区间均完整落在劣势一侧。这说明当前GRU学到了账号特有的发帖水平和状态尺度，不能把
Elon权重直接用于其他账号，也不能据此接入生产。

保留的实验目录：

- `experiments/generalization/whitehouse/20261017T164308.663628Z`：WhiteHouse重训。
- `experiments/generalization/whitehouse-zero-shot/20261017T165411.240272Z`：Elon权重
  zero-shot迁移，清单内记录了所有源权重SHA-256。

可用同一源实验复现zero-shot测试：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv-gpu\Scripts\elon-model.exe --handle WhiteHouse experiment-gru-transfer `
  --source-run experiments/gru-v4/20261017T163347.179620Z `
  --event-start 2026-10-16T16:00:00Z `
  --event-end 2026-10-23T16:00:00Z `
  --interval-config config/whitehouse-seven-day-generalization-intervals.json
```

迁移命令只读取源模型并推理，不在目标账号上做反向传播、温度选择、集成权重选择或模型
选择。目标账号的验证和测试标签仅用于评分；目标历史中观察时点之前的帖子仍会作为输入
序列，这是预测时必需的可用信息。

#### 当前盘口四模型比较

`forecast-gru-comparison`使用同一观察整点比较路由基线、原始GRU-NLL、V3密集小时GRU
和V4校准集成。路由模型用目标盘口开盘前的完整历史重新拟合；三种GRU使用各自最新的
严格样本外fold-2 checkpoint，其训练标签截止2026年8月7日。程序会记录源权重哈希、
官方累计数与数据库计数差、完整26档概率、80%区间和最佳连续三档，不会训练或下单。

2026年10月18日北京时间01:02的固定快照结果如下。模型输入截到完整的01:00整点，
xTracker官方累计数与MySQL可见帖子数完全一致。

| 13–20日盘口（当前253，剩余71小时） | 预测均值 | 80%区间 | 最佳连续三档 | 三档概率 |
| --- | ---: | ---: | --- | ---: |
| 路由基线（GBM-84h） | 408.1 | 361–463 | 360–379 / 380–399 / 400–419 | 59.2% |
| 原始GRU-NLL | 408.5 | 345–480 | 360–379 / 380–399 / 400–419 | 44.3% |
| V3密集小时GRU | 393.2 | 329–466 | 340–359 / 360–379 / 380–399 | 44.1% |
| V4校准集成 | 397.7 | 327–479 | 340–359 / 360–379 / 380–399 | 40.1% |

| 16–23日盘口（当前48，剩余143小时） | 预测均值 | 80%区间 | 最佳连续三档 | 三档概率 |
| --- | ---: | ---: | --- | ---: |
| 路由基线（GBM-24h） | 312.1 | 252–394 | 260–279 / 280–299 / 300–319 | 48.5% |
| 原始GRU-NLL | 329.2 | 233–434 | 280–299 / 300–319 / 320–339 | 30.4% |
| V3密集小时GRU | 317.6 | 225–419 | 260–279 / 280–299 / 300–319 | 31.3% |
| V4校准集成 | 317.2 | 215–431 | 260–279 / 280–299 / 300–319 | 28.6% |

完整快照保存在：

- `experiments/live-comparisons/20261017T1702Z/oct-13-20.json`
- `experiments/live-comparisons/20261017T1702Z/oct-16-23.json`

复现单个盘口的命令示例：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv-gpu\Scripts\elon-model.exe forecast-gru-comparison `
  --event-start 2026-10-13T16:00:00Z `
  --event-end 2026-10-20T16:00:00Z `
  --as-of 2026-10-17T17:02:44Z `
  --current-count 253 `
  --interval-config config/elon-musk-oct-13-20-2026-intervals.json `
  --output experiments/live-comparisons/20261017T1702Z/oct-13-20.json
```

这些概率是模型分布，不是市场价格优势。是否建仓仍需另外比较手续费、滑点后的可成交
价格；四个模型共享大量历史和特征，也不能把四票一致当作四个独立证据。

GPU训练使用独立环境。RTX 3060及当前驱动可使用CUDA 13.0 wheel：

```powershell
# 只需首次执行
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install `
  "torch==2.14.1+cu130" `
  --index-url https://download.pytorch.org/whl/cu130
.\.venv-gpu\Scripts\python.exe -m pip install -e ".[dev,ml]"

# 使用盘口开始前的数据训练，最终测试不会参与早停或选择
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\.venv-gpu\Scripts\elon-model.exe experiment-gru
```

每次运行写入新的 `experiments/gru/<时间戳>/`，其中包括：

- `data-quality.json` 和 `cleaned-hourly-counts.jsonl`：清洗结果与小时序列。
- `dataset-index.jsonl`：每个fold、事件和观察时点的标签索引。
- `fold-*/**.pt`：模型权重、特征缩放参数、随机种子和最佳epoch。
- `predictions.jsonl`：逐样本完整区间概率。
- `metrics.json`：按fold和观察时点拆分的概率、覆盖率与稳定性指标。
- `comparisons-vs-route.json`：相对冻结路由的事件级配对bootstrap区间。
- `manifest.json`：完整参数、CUDA设备、时间边界与数据限制。

这项实验使用清洗后的完整历史时间戳，尚不能完全还原帖子当时是否已被xTracker捕获，
因此结果仍属于理想历史口径。只有未来真实盘口验证通过后，才会新增
`forecast_strategy=gru`；目前实时系统继续使用 `route` 和 `--observe-only`。

数据对账命令会重新读取同一追踪周期，检测端点遗漏和疑似删除，但不会从 MySQL 删除
帖子：

```powershell
.\.venv\Scripts\elon-data.exe reconcile `
  --start 2026-10-13T16:00:00Z `
  --end 2026-10-20T16:00:00Z
```

首次对账结果为官方 tracking `242`、posts 端点 `242`、MySQL `242`，差值为0，未发现
疑似删除。结果保存在 `xtracker_reconciliation_runs`，并嵌入后续实验 manifest。

每轮会读取所有区间的最高买价、最低卖价和订单簿深度，并在计入 Polymarket 动态
taker 手续费后计算模型概率优势、组合可平仓价值和候选滚仓成本。当前这些结果只用于
观察与报警；单区间的旧版买卖决策器不代表最终的三档策略。

先执行一次历史回测和只读评估：

```powershell
.\.venv\Scripts\elon-arb.exe --once --observe-only --backtest
```

持续只读监控应使用脚本启动。不要直接运行不带 `--observe-only` 的 `elon-arb`，因为
三档建仓与持久化滚仓确认尚未完成：

```powershell
.\scripts\start_elon_monitor.ps1
```

只验证数据、模型、盘口和报警链路，不记录或提交订单：

```powershell
.\.venv\Scripts\elon-arb.exe --once --observe-only
```

后台持续运行或停止两分钟监控：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\scripts\start_elon_monitor.ps1
.\scripts\stop_elon_monitor.ps1
```

后台进程PID保存在 `elon-monitor.pid`，标准输出和错误日志分别写入
`elon-monitor.stdout.log`、`elon-monitor.stderr.log`。

同时监控多个盘口时，必须为每个盘口隔离PID、状态、审计和日志。通用脚本会将文件放到
`runtime/monitors/<名称>-*`，并强制使用两分钟轮询、路由模型和只读模式。脚本会启用
四模型区间变化提醒，并从 `models/` 加载已经封存的样本外fold-2权重：

```powershell
$env:MYSQL_PASSWORD = "你的本机MySQL密码"
.\scripts\start_elon_event_monitor.ps1 `
  -Name "oct-13-20" `
  -EventSlug "elon-musk-of-tweets-october-13-october-20-2026"
.\scripts\start_elon_event_monitor.ps1 `
  -Name "oct-16-23" `
  -EventSlug "elon-musk-of-tweets-october-16-october-23-2026"

.\scripts\stop_elon_event_monitor.ps1 -Name "oct-13-20"
.\scripts\stop_elon_event_monitor.ps1 -Name "oct-16-23"
```

每轮监控同时写入 MySQL：

- `elon_monitor_snapshots`：累计数量、实时状态、预测、最佳三档、持仓价值和报警。
- `polymarket_quote_snapshots`：每个区间的可成交 Bid/Ask、手续费后价格和模型概率。
- `polymarket_orderbook_levels`：每个区间最靠近成交价的10档买卖深度，用于市价单
  滑点和滚仓成本回放。

所有区间通过官方批量订单簿接口一次获取。单边盘口仍会进入概率计算，但没有挂单的
方向按不可成交处理，不会把页面显示价误当作真实平仓价格。

配置钉钉机器人后，区间移动、发帖状态切换、每增加5个百分点的峰值回撤、相对首次
可平仓价值恶化12%，以及总亏损达到25%都会报警。Webhook和签名密钥只放环境变量：

```powershell
$env:DINGTALK_WEBHOOK = "你的机器人Webhook"
$env:DINGTALK_SECRET = "你的签名密钥"
```

程序每两分钟计算一次，但不会每两分钟发送无变化的状态。钉钉仅在以下事件发生时推送：

- 路由模型切换，例如从 GBM-24 切换到 GBM-42。
- 最佳连续三档发生变化，并附新三档概率；有持仓时同时附换仓价值改善。
- 同一条区间变化提醒还会给出路由基线、原GRU-NLL、V3和V4的最终均值、80%区间、
  最佳连续三档、各档概率与三档合计概率。四模型计算失败不会吞掉主提醒，而会标明原因。
- 发帖状态在 extreme-low / low / normal / high / extreme-high 之间切换。
- 持仓从峰值每增加5个百分点回撤、相对首次可平仓价值下降12%，或组合亏损达到25%。
- 盘口结束并经过标签确认等待期后，记录官方最终数量和命中区间；官方总数变化时发送
  标签修订通知。

盘口最终标签保存在 `elon_event_outcomes`。默认在结束60分钟后使用 xTracker tracking
总数作为权威标签，同时保存 posts 端点和 MySQL 数量用于对账。相同标签重复观察不会
重复报警；最终数量或命中区间变化时 `label_version` 递增。若监控在结算时没有运行，
可以事后补录：

```powershell
.\.venv\Scripts\elon-data.exe finalize-event `
  --start 2026-10-16T16:00:00Z `
  --end 2026-10-23T16:00:00Z `
  --interval-config config\elon-musk-oct-16-23-2026-intervals.json
```

事件标签通过 `event_slug` 与 `elon_monitor_snapshots`、`polymarket_quote_snapshots` 关联：
开盘附近快照用于检验和迭代 P0，事件内所有快照用于检验实时修正器。标签进入下一轮
时序训练和封存测试，但单个新盘口不会自动覆盖生产模型；只有新模型通过 log loss、
Brier、ECE、三档命中率和交易回放门槛后才允许晋级。

当前事件的默认配置是 `elon-musk-of-tweets-october-13-october-20-2026`。历史缓存写入
`elon-history-cache.json`，旧版非只读模式的纸面持仓写入 `elon-arb-state.json`，每轮
模型、报价和决策写入 `elon-arb-audit.jsonl`。这些运行文件都不会提交到 Git。

主要参数：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `ELON_HISTORY_DAYS` | `365` | 训练历史长度 |
| `ELON_MIN_ENTRY_EDGE` | `0.035` | 扣除买入手续费后的最低概率优势 |
| `ELON_MIN_EXIT_ADVANTAGE` | `0.015` | 卖出净价超过模型概率的最低幅度 |
| `ELON_ORDER_AMOUNT_USD` | `20` | 每档默认投入金额 |
| `ELON_MAX_TOTAL_COST_USD` | `100` | 当前组合成本上限 |
| `ELON_MAX_DAILY_BUY_USD` | `100` | UTC 单日买入上限 |
| `ELON_MAX_POSITIONS` | `3` | 最大同时持仓区间数 |
| `ELON_POLL_INTERVAL_SECONDS` | `120` | xTracker与盘口默认轮询周期 |
| `ELON_MIN_ENTRY_ELAPSED_MINUTES` | `1440` | 盘口开始后最早入场时间 |
| `ELON_DRAWDOWN_ALERT_STEP` | `0.05` | 峰值回撤报警步长 |
| `ELON_DETERIORATION_WARNING` | `0.12` | 相对初始可平仓价值恶化预警 |
| `ELON_MAX_STRATEGY_LOSS` | `0.25` | 组合最大亏损预警 |
| `ELON_LABEL_FINALIZATION_DELAY_MINUTES` | `60` | 盘口结束后等待多久确认最终标签 |
| `ELON_FOUR_MODEL_ALERTS` | `false` | 三档变化时是否追加四模型预测；多盘口启停脚本会设为true |
| `ELON_GRU_DEVICE` | `auto` | GRU推理设备；Windows NVIDIA自动用CUDA，macOS自动用CPU |
| `ELON_FOUR_MODEL_TIMEOUT_SECONDS` | `120` | 四模型提醒计算超时 |
| `ELON_MIN_BACKTEST_COVERAGE` | `0.65` | 实盘所需的最低 80% 区间覆盖率 |
| `ELON_MAX_BACKTEST_MAE` | `50` | 实盘允许的最大回测平均绝对误差 |

代码保留了私钥、钱包地址、双重开关、回测门槛和地域限制检查等真实下单防线，但这
不代表当前三档策略已经可以实盘。完成持久化滚仓状态机、纸面仓位导入和基于历史深度
的收益回测以前，`ELON_ENABLE_LIVE` 必须保持 `false`。私钥不要写入源码或提交到 Git。

源码入口：

- `src/polymarket_bot/elon_main.py`：命令行和持续轮询。
- `src/polymarket_bot/elon_trader.py`：市场发现、报价、仓位、下单和审计。
- `src/polymarket_bot/forecast.py`：经验窗口模型和走步回测。
- `src/polymarket_bot/elon_model.py`：MySQL数据审计、训练报告和实时预测入口。
- `src/polymarket_bot/elon_monitor.py`：连续三档选择、持仓估值和滚仓成本评估。
- `src/polymarket_bot/notifier.py`：钉钉机器人签名与报警发送。
- `src/polymarket_bot/elon_strategy.py`：手续费、入场和卖出判断。
- `src/polymarket_bot/xtracker.py`：xTracker API 与历史缓存。
- `src/polymarket_bot/elon_config.py`：全部环境变量和风控配置。

## 固定阈值机器人

策略很直接：所选结果的可执行买价不高于 `BUY_BELOW` 时买入；已有仓位的
可执行卖价达到 `TAKE_PROFIT_AT` 或跌到 `STOP_LOSS_AT` 时全部卖出。机器人还会
限制单笔金额、最大持仓、每日累计买入、滑点和交易频率。

## 安装

需要 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

### macOS

可以运行。Python业务代码、MySQL访问、xTracker同步、路由模型、GRU推理和钉钉提醒都
是跨平台的；Windows的 `.ps1` 不能直接在macOS的默认Shell中执行，因此仓库同时提供
了 `.sh` 启停脚本。Apple Silicon没有CUDA，`ELON_GRU_DEVICE=auto` 会使用CPU；当前小型
GRU只在区间变化时运行，CPU推理足够，训练会比NVIDIA显卡慢。

```bash
git clone https://github.com/NieMinxin/elon-.git
cd elon-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,ml]'

# MySQL可以使用本机服务或Docker；先创建polymarket_bot数据库。
export MYSQL_PASSWORD='你的本机MySQL密码'
export DINGTALK_WEBHOOK='你的机器人Webhook'
export DINGTALK_SECRET='你的签名密钥'

./scripts/start_elon_event_monitor.sh \
  oct-16-23 elon-musk-of-tweets-october-16-october-23-2026
./scripts/stop_elon_event_monitor.sh oct-16-23
```

Intel Mac与Apple Silicon都可以使用同一命令。若PyTorch在较老的Intel macOS版本上没有
对应wheel，可先安装不含`ml`的依赖运行路由监控，并关闭`ELON_FOUR_MODEL_ALERTS`；四模型
提醒需要PyTorch和仓库中 `models/` 的封存权重。

PowerShell 不会自动读取 `.env`。测试时可以逐项设置当前终端的环境变量：

```powershell
$env:POLYMARKET_MARKET_SLUG = "你的市场-slug"
$env:POLYMARKET_OUTCOME = "YES"
$env:POLYMARKET_BUY_BELOW = "0.40"
$env:POLYMARKET_TAKE_PROFIT_AT = "0.60"
$env:POLYMARKET_STOP_LOSS_AT = "0.25"
```

市场 slug 是 `https://polymarket.com/market/<slug>` 中最后一段。`OUTCOME` 可设为
`YES` 或 `NO`。

## 先运行纸面交易

只检查一次：

```powershell
polymarket-bot --once
```

持续轮询：

```powershell
polymarket-bot
```

纸面仓位和每日买入额保存在 `bot-state.json`，因此重启不会立即重复模拟买入。

## 开启真实交易

先在 Polymarket 网页端确认账户的钱包地址，并确保钱包有 pUSD。私钥只能放在
环境变量或密钥管理器中，不能写进代码、`.env.example` 或 Git。

```powershell
$env:POLYMARKET_PRIVATE_KEY = "0x..."
$env:POLYMARKET_WALLET_ADDRESS = "0x..."
$env:POLYMARKET_ENABLE_LIVE = "true"
polymarket-bot --live --once
```

确认单次运行符合预期后，去掉 `--once` 才会持续自动交易。真实订单必须同时满足
`POLYMARKET_ENABLE_LIVE=true` 和命令行 `--live`，缺少任意一个都会停留在纸面模式
或直接拒绝启动。

机器人提交 FAK 市价单：能成交的部分立即成交，剩余部分取消。买入使用
`max_spend` 和 `max_price`，卖出使用 `min_price`，以限制手续费预算和最差成交价。
每笔真实订单前还会调用官方 geoblock 接口；受限地区不会提交订单。

## 配置

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `POLYMARKET_MARKET_SLUG` | 必填 | 单个二元市场 slug |
| `POLYMARKET_OUTCOME` | `YES` | 交易 `YES` 或 `NO` |
| `POLYMARKET_BUY_BELOW` | `0.40` | 触发买入的最高可执行价格 |
| `POLYMARKET_TAKE_PROFIT_AT` | `0.60` | 触发止盈的最低卖价 |
| `POLYMARKET_STOP_LOSS_AT` | `0.25` | 触发止损的最高卖价 |
| `POLYMARKET_ORDER_AMOUNT_USD` | `10` | 每笔最多使用的 pUSD |
| `POLYMARKET_MAX_POSITION_SHARES` | `50` | 所选结果最大份额 |
| `POLYMARKET_MAX_DAILY_BUY_USD` | `25` | UTC 自然日累计买入上限 |
| `POLYMARKET_MAX_SLIPPAGE` | `0.02` | 最差成交价相对当前价的偏离 |
| `POLYMARKET_POLL_INTERVAL_SECONDS` | `30` | 轮询间隔，最少 5 秒 |
| `POLYMARKET_TRADE_COOLDOWN_SECONDS` | `300` | 两次交易之间的冷却时间 |
| `POLYMARKET_STATE_PATH` | `bot-state.json` | 本地状态文件 |

价格阈值不是收益保证。薄盘口、延迟、部分成交、费用、市场暂停和链上结算失败都可能
让真实结果偏离当前报价。先用小额纸面测试，并遵守所在地法律和 Polymarket 条款。

## 开发验证

```powershell
pytest
ruff check .
```

实现依据：

- [Polymarket Predictions API](https://docs.polymarket.com/api-reference/predictions/overview)
- [官方 Python SDK](https://docs.polymarket.com/getting-started/python)
- [下单文档](https://docs.polymarket.com/trading/place-orders)
- [地域限制](https://docs.polymarket.com/api-reference/geoblock)

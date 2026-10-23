# option_calculation
# 可转债相对估值：近邻拟合 + 自身历史 + 风险预警

参考中金《转债"低估"策略优化与 Python 实现》（2021-04-19）公开的方法：对每只转债找属性相近的同类券，用同类券拟合"平价 → 转股溢价率"曲线，算出合理价格，再看实际价格偏离多少。

三个脚本逐步叠加功能，只用 Python 标准库，不依赖 pandas、NumPy、BSM 或二叉树模型。

| 文件 | 功能 | 输入 |
|---|---|---|
| `option_calcu.py` | 仅用截面数据建模 | 内置示意数据 |
| `option_calcu_hist.py` | 截面数据 + 自身历史数据 | 当天行情 + 残差历史 |
| `option_calcu_hist_risk.py` | 在第二个基础上增加 ST / 退市风险预警 | 当天行情 + 历史行情（均含风险字段） |

> 仓库里的 CSV 均为示意数据，不是真实行情。

---

## 1. option_calcu.py：仅用截面数据

### 代码思路

```
目标转债CB01
    ↓
根据发行规模、分析师覆盖、趋势等特征
    ↓
neighborsCbGroups找出相似转债
    ↓
取得这些相似转债的平价和溢价率
    ↓
olsForGroup拟合曲线
    ↓
把CB01的平价代入曲线
    ↓
得到CB01的合理溢价率和合理价格
    ↓
实际价格与合理价格比较
    ↓
diff = 实际价格 - 合理价格
    ↓
对 CB02、CB03……所有转债重复以上过程
    ↓
将所有转债的 diff 标准化为 z_score
    ↓
按照 z_score 从小到大排序
    ↓
获得全市场相对低估排序
    ↓
（人工）加入信用、退市、强赎、流动性和趋势过滤
    ↓
（人工）选择排名靠前的一组转债构建组合
```

### 运行

```bash
python3 option_calcu.py               # 内置 12 只示意券的单日估值
```

---

## 2. option_calcu_hist.py：截面数据 + 自身历史数据

### 代码思路

```
读取今天的行情 today.csv（--input）
    ↓
enrich_market_fields 补齐平价、溢价率、价格
    ↓
目标转债CB01
    ↓
根据发行规模、分析师覆盖、趋势等特征
    ↓
neighborsCbGroups找出相似转债
    ↓
取得这些相似转债的平价和溢价率
    ↓
olsForGroup拟合曲线
    ↓
把CB01的平价代入曲线，得到合理溢价率和合理价格
    ↓
residual_pct = 实际溢价率 - 拟合溢价率（百分点，存入历史用）
diff = 实际价格 - 合理价格（元）
    ↓
对 CB02、CB03……所有转债重复以上过程
    ↓
将所有转债的 diff 标准化为 z_score，从小到大排序
（截面结果：和同类券比，今天贵不贵）
    ↓
读取残差历史 history.csv（--history）
    ↓
对每只转债，取今天之前最近 60 / 120 个交易日的残差
    ↓
计算今天的残差在自身历史里的位置：
pct60 / pct120 = 分位（0 = 比过去每天都便宜，100 = 最贵）
selfz60 = 偏离自身常态几个标准差
norm60 = 过去 60 日平均残差（这只券的"常态"）
（历史结果：和自己比，今天贵不贵）
    ↓
把今天每只转债的残差写入 history.csv（同一天重跑会覆盖）
    ↓
打印：resid_pp、z（和同类比）+ pct60、pct120、selfz60、norm60（和自己比）
    ↓
两个口径组合判断：
和同类比便宜 + 和自己比也便宜 → 两头都便宜，重点候选
和同类比便宜 + 和自己比正常   → 一直便宜，先查原因
和同类比正常 + 和自己比便宜   → 相对自身变便宜，观察
和同类比贵   + 和自己比也贵   → 两头都贵，回避
    ↓
（人工）排除信用、退市、强赎、流动性风险后选券
```

### 运行

```bash
python3 option_calcu_hist.py --input today.csv --date 2026-10-23 --history history.csv
```

### 输入数据

**today.csv**：当天行情，每行一只券。

```
code,conv,prem,close,issue_size,analyst_coverage,momentum_60d
CB01,80.86,43.9,116.364,24.0,1.0,-10.0
```

**history.csv**：残差历史，由程序每天运行后自动写入，第一次运行时可以不存在。

```
date,code,parity,actual_premium_pct,fitted_premium_pct,residual_pct,diff,diff_pct,z_score
2026-04-17,CB01,85.420000,44.250000,39.533948,4.716052,4.027102,3.378727,1.248831
```

---

## 3. option_calcu_hist_risk.py：截面 + 自身历史 + ST / 退市风险预警

### 代码思路

```
读取今天的行情 today_withST.csv（--input，含 ST、退市、债底、正股价等风险字段）
    ↓
读取历史行情 history_withST.csv（--history-data，同样的列 + date，每天每只券一行）
    ↓
【风险筛查】screen_risky（加 --exclude-risky 时生效）
剔除：正股 ST、已锁定退市、正股价 < 1.2 元、停牌、已公告强赎、价格 < 债底的 95%
打印剔除名单和原因；被剔除的券不参与找近邻、拟合和 z 值
    ↓
【重算历史残差】residuals_from_history
对历史行情最近 120 个交易日，每天用同样的筛查规则和同样的特征
重新做一遍截面估值，得到每只转债每天的残差
（历史残差始终和今天的算法一致，不需要单独保存残差文件）
    ↓
【风险提醒】risk_watch：看每只转债自己（包括被剔除的券）的历史走势
高：价格低于债底、正股连续低于 1 元（显示面值退市倒计时）、正股 ST、已锁定退市
中：价格距债底不足 5%、正股价 < 1.5 元、平价 60 日跌超 20%、转债价格 20 日跌超 8%、到期收益率偏高或快速上升
提示：平价不在近邻的平价范围内，估值是外推的，仅供参考
    ↓
【截面估值】对筛查后剩下的转债：
目标转债 → neighborsCbGroups 找相似转债 → olsForGroup 拟合曲线
→ 合理溢价率、合理价格 → residual_pct、diff → 所有转债 diff 标准化为 z_score 并排序
    ↓
【自身历史比较】今天的残差 vs 重算出的历史残差
→ pct60 / pct120 / selfz60 / norm60
    ↓
把今天的行情追加写入 history_withST.csv（同一天重跑会覆盖），明天接着用
    ↓
依次打印：风险筛查结果 → 风险提醒 → 估值表
若有"高"风险但没开筛查，表的最后额外提醒
    ↓
综合判断：
先看风险提醒，"高"直接回避，"中"人工核查，"提示"的估值数字仅供参考
再按截面（z）和自身历史（pct、selfz）两个口径判断贵贱
    ↓
（人工）选择排名靠前、且没有风险提醒的一组转债构建组合
```

### 运行

```bash
# 开启风险筛查：剔除高风险券，并打印风险提醒
python3 option_calcu_hist_risk.py --input today_withST.csv --date 2026-10-23 --history-data history_withST.csv --exclude-risky

# 不剔除，只提醒
python3 option_calcu_hist_risk.py --input today_withST.csv --date 2026-10-23 --history-data history_withST.csv
```

### 输入数据

**today_withST.csv**：当天行情，每行一只券，含风险字段。

```
code,conv,prem,close,stock_price,conversion_price,bond_floor,is_st,delisting,suspended,issue_size,analyst_coverage,momentum_60d
CB13,42.58,60.74,68.44,0.85,2.0,92.0,1,1,0,6.0,0.0,-55.0
```

**history_withST.csv**：历史行情，列和当天行情相同，多一个 `date` 列，每天每只券一行。程序每次运行后会把当天行情追加进去。

```
date,code,conv,prem,close,stock_price,conversion_price,bond_floor,is_st,delisting,suspended,issue_size,analyst_coverage,momentum_60d
2026-04-17,CB01,85.91,43.23,123.048,6.87,8,96,0,0,0,24.0,1.0,-10.0
```

---

## 字段说明

| 字段 | 含义 |
|---|---|
| `code` | 转债代码 |
| `conv` | 平价（转股价值）= 正股价 ÷ 转股价 × 100 |
| `prem` | 转股溢价率（%）= 转债价格 ÷ 平价 − 1 |
| `close` | 转债收盘价 |
| `issue_size` | 发行规模（亿元），找近邻时取对数 |
| `analyst_coverage` | 分析师覆盖家数 |
| `momentum_60d` | 正股过去 60 日涨跌幅（%） |
| `stock_price` | 正股价 |
| `conversion_price` | 转股价 |
| `bond_floor` | 纯债价值（债底） |
| `is_st` | 正股是否 ST，1 是 0 否 |
| `delisting` | 是否进入退市整理期或已锁定退市，1 是 0 否 |
| `suspended` | 是否停牌，1 是 0 否 |
| `ytm`（可选） | 到期收益率（%），有这一列时启用到期收益率预警 |
| `call_announced`（可选） | 是否已公告强赎，1 是 0 否 |

## 输出指标

| 指标 | 含义 | 怎么读 |
|---|---|---|
| `resid_pp` | 实际溢价率 − 近邻拟合溢价率（百分点） | 负数比同类便宜，正数比同类贵 |
| `z` | diff 在全部券中的标准化分数 | 绝对值 > 2 为明显异常 |
| `pct60` / `pct120` | 今天残差在自身过去 60 / 120 日中的分位 | 越低越比自己平时便宜 |
| `selfz60` | 偏离自身常态几个标准差 | 负数低于常态 |
| `norm60` | 过去 60 日平均残差 | 这只券平时比同类贵还是便宜 |

## 注意事项

- 估值结果只说明相对贵贱，不说明原因，也不保证价格回归；候选券需要逐只核查信用、强赎、下修和流动性。
- 平价落在近邻平价范围之外时，三参数曲线是外推的，结果不稳定。
- 风险规则的阈值为示意值，可在 `screen_risky` 的参数和 `RiskWatchConfig` 中调整。
- 每次运行都会写入当天数据，同一天重复运行会覆盖。用 Wind 导出的 GBK 编码 CSV 时加 `--encoding gbk`。

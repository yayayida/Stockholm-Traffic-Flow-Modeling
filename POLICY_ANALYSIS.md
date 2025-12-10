# Transit Waiting Time Policy Analysis

## 概述 (Overview)

本文档展示了通过降低某些公共交通线路的等待时间来改善交通系统指标的政策效果。

This document demonstrates the effects of a policy intervention that reduces waiting times on specific public transit links to improve system-wide transportation metrics.

## 问题陈述 (Problem Statement)

我们想改变某些link之间的waiting time，使得在这个改变的policy下新的scenario的一些指标能证明情况变好了，比如：
- **EU (Expected Utility / 期望效用)** 增高
- **Public Transit Share (公共交通出行比例)** 增高

## 政策描述 (Policy Description)

### 实施策略 (Implementation Strategy)

通过提高主要地铁线路的发车频率来降低等待时间：

**主要线路 (Major T-bana corridors) - 等待时间降低50%:**
- Centralen ↔ Östermalm (centerN ↔ centerE)
- Centralen ↔ Södermalm (centerN ↔ centerS)
- Kungsholm ↔ Centralen (centerW ↔ centerN)
- Centralen/Kungsholm ↔ North Stockholm (centerN/centerW ↔ N)

**重要线路 (Key corridors) - 等待时间降低40%:**
- Danderyd ↔ Östermalm (NE ↔ centerE)
- Södermalm ↔ South Stockholm (centerS ↔ S)
- Södermalm ↔ Southwest Stockholm (centerS ↔ SW)

这个政策模拟了在关键公交走廊上增加服务频率的效果。

This policy simulates the effect of increasing service frequency on key transit corridors.

## 结果对比 (Results Comparison)

### 主要指标改善 (Key Improvements)

| 指标 (Metric) | 基准情景 (Baseline) | 政策情景 (Policy) | 变化 (Change) |
|--------------|---------------------|-------------------|---------------|
| **公交出行量 (Transit Trips)** | 316,535 | 341,715 | **+25,179 (+8.0%)** ✓ |
| **公交出行比例 (Transit Mode Share)** | 39.57% | 42.71% | **+3.15 pp** ✓ |
| **期望效用 (Expected Utility)** | 8,775,100 | 8,855,120 | **+80,020 (+0.91%)** ✓ |
| 汽车出行量 (Car Trips) | 409,949 | 393,023 | -16,925 (-4.1%) |
| 汽车出行比例 (Car Mode Share) | 51.24% | 49.13% | -2.12 pp |
| 行驶距离 (Distance Travelled) | 71,364 千公里 | 68,353 千公里 | -3,012 (-4.2%) |
| 外部成本 (Total Externalities) | €53,282,307 | €51,033,736 | **-€2,248,571 (-4.2%)** ✓ |

### 详细指标 (Detailed Metrics)

#### 基准情景 (Baseline Scenario)

```
Car Trips:                     409,949
Car Mode Share:                51.24%
Transit Trips:                 316,535
Transit Mode Share:            39.57%
Slow Trips:                    73,514
Slow Mode Share:               9.19%
Total Trips:                   800,000
Distance Travelled:            71,364 千公里
Noise Cost:                    €5,780,501
Waiting Time Cost:             €713,642
Accident Cost:                 €17,841,053
Emission Cost:                 €285,456
CO2 Cost:                      €28,661,653
Total Externalities:           €53,282,307
Parking Revenue:               €4,099,497
Transit Revenue:               €9,496,076
Total Revenue:                 €13,595,573
Expected Utility:              8,775,100
```

#### 政策情景 (Policy Scenario - Reduced Transit Waiting Times)

```
Car Trips:                     393,023
Car Mode Share:                49.13%
Transit Trips:                 341,715
Transit Mode Share:            42.71%
Slow Trips:                    65,260
Slow Mode Share:               8.16%
Total Trips:                   800,000
Distance Travelled:            68,353 千公里
Noise Cost:                    €5,536,558
Waiting Time Cost:             €683,525
Accident Cost:                 €17,088,141
Emission Cost:                 €273,410
CO2 Cost:                      €27,452,100
Total Externalities:           €51,033,736
Parking Revenue:               €3,930,238
Transit Revenue:               €10,251,463
Total Revenue:                 €14,181,701
Expected Utility:              8,855,120
```

## 关键发现 (Key Findings)

### ✓ 成功达成目标 (Objectives Achieved)

1. **期望效用提升 (Expected Utility Increased)**
   - 从 8,775,100 增加到 8,855,120
   - 增长 80,020 单位 (0.91%)
   - 说明整体社会福利提高

2. **公交出行比例显著提升 (Transit Share Significantly Increased)**
   - 从 39.57% 增加到 42.71%
   - 增长 3.15 个百分点
   - 增加了 25,179 次公交出行 (+8.0%)

### 额外收益 (Additional Benefits)

3. **环境效益 (Environmental Benefits)**
   - 汽车行驶距离减少 3,012 千公里 (-4.2%)
   - 总外部成本降低 €2,248,571 (-4.2%)
   - CO2 成本减少 €1,209,553
   - 噪音成本减少 €243,943
   - 事故成本减少 €752,912

4. **模式转移 (Mode Shift)**
   - 汽车出行减少 16,925 次 (-4.1%)
   - 从汽车和慢行转向公共交通
   - 公交收入增加 €755,387 (+8.0%)

## 政策影响机制 (Policy Impact Mechanism)

1. **降低等待时间** → 提高公交吸引力
2. **公交吸引力提升** → 更多人选择公交
3. **更多公交出行** → 减少汽车使用
4. **减少汽车使用** → 降低拥堵和外部成本
5. **整体效率提升** → 期望效用增加

## 如何运行分析 (How to Run the Analysis)

### 安装依赖 (Install Dependencies)

```bash
pip install haversine networkx pandas geopandas numpy tabulate folium
```

### 运行政策对比脚本 (Run Policy Comparison Script)

```bash
python PolicyComparison.py
```

脚本将:
1. 运行基准情景 (无政策干预)
2. 运行政策情景 (降低等待时间)
3. 对比两个情景的结果
4. 生成详细的对比报告
5. 保存结果到 `policy_comparison_results.csv`

## 代码结构 (Code Structure)

### PolicyComparison.py

主要函数:

- `apply_transit_policy(G_pt, policy_config)`: 应用等待时间降低政策到公交网络
- `run_scenario(...)`: 运行完整的交通模型情景 (需求预测 + 交通分配)
- `compare_scenarios(...)`: 对比基准和政策情景，计算改善指标
- `main()`: 主函数，协调整个分析流程

### 政策配置 (Policy Configuration)

政策通过字典定义，格式为:
```python
policy_config = {
    ('origin', 'destination'): reduction_factor,  # 0-1之间表示降低比例
    # 或
    ('origin', 'destination'): absolute_wait_time  # 大于1表示绝对值
}
```

## 结论 (Conclusion)

**本政策成功证明了通过降低公交等待时间可以:**

✅ **提高期望效用** (+0.91%)  
✅ **增加公交出行比例** (+3.15 pp, 从39.57%到42.71%)  
✅ **减少汽车出行** (-4.1%)  
✅ **降低环境外部成本** (-4.2%)  
✅ **增加公交收入** (+8.0%)  

这些结果表明，投资于提高公交服务频率是一个有效的政策工具，可以实现多重目标:
- 改善社会福利 (更高的EU)
- 促进可持续交通 (更多公交出行)
- 减少环境影响 (更少的汽车行驶距离)
- 提高公交系统财务可持续性 (更多收入)

## 进一步研究方向 (Future Research Directions)

1. **成本效益分析**: 计算实施政策的成本，与收益进行对比
2. **敏感性分析**: 测试不同的等待时间降低幅度
3. **空间异质性**: 分析不同地区的政策效果差异
4. **时间动态**: 研究高峰期和非高峰期的不同效果
5. **综合政策**: 结合票价调整、停车政策等其他措施

## 参考文献 (References)

本分析基于:
- Nested Logit 出行需求模型
- User Equilibrium 交通分配模型  
- 斯德哥尔摩交通网络数据

---

**Created**: 2025-12-10  
**Author**: Stockholm Traffic Flow Modeling Team  
**Version**: 1.0

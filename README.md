# Stockholm-Traffic-Flow-Modeling

Analyzing traffic flow patterns and different scenarios in Stockholm

## Overview

This repository contains a transportation demand and assignment model for Stockholm, Sweden. The model simulates:
- Multi-modal travel demand (car, public transit, slow modes)
- Traffic assignment using User Equilibrium
- Network performance metrics and externalities

## New Feature: Transit Policy Analysis

**`PolicyComparison.py`** - A comprehensive script that demonstrates how transit service improvements can enhance system-wide performance.

### Policy Intervention
The script implements a policy to reduce waiting times on major transit corridors:
- **50% reduction** on core T-bana (metro) lines connecting to Centralen
- **40% reduction** on key suburban connections

### Results Summary

The policy intervention successfully demonstrates:

| Metric | Baseline | Policy | Improvement |
|--------|----------|--------|-------------|
| **Expected Utility** | 8,775,100 | 8,855,120 | **+0.91%** ✓ |
| **Transit Share** | 39.57% | 42.71% | **+3.15 pp** ✓ |
| Transit Trips | 316,535 | 341,715 | +8.0% |
| Car Trips | 409,949 | 393,023 | -4.1% |
| Total Externalities | €53.3M | €51.0M | **-4.2%** |

See **[POLICY_ANALYSIS.md](POLICY_ANALYSIS.md)** for detailed results and analysis (中英文双语).

### How to Run

```bash
# Install dependencies
pip install haversine networkx pandas geopandas numpy tabulate folium

# Run the policy comparison
python PolicyComparison.py
```

Output includes:
- Console output with detailed iteration logs
- Comparison tables showing baseline vs policy results
- `policy_comparison_results.csv` - Summary metrics in CSV format

## Project Structure

- `Network.py` - Road and transit network definitions
- `Landuse.py` - Land use and zone data loading
- `Assignment.py` - Traffic assignment algorithms (User Equilibrium)
- `Effect.py` - Performance metrics and externalities calculation
- `Visualization.py` - Network visualization tools
- `MainModel.ipynb` - Main Jupyter notebook for model execution
- **`PolicyComparison.py`** - Transit policy scenario comparison (NEW)
- **`POLICY_ANALYSIS.md`** - Detailed policy analysis documentation (NEW)

## Model Components

### Demand Model
- Nested Logit model with car ownership
- Mode choice (car, transit, slow modes)
- Destination choice based on employment

### Assignment Models
- **Car**: User Equilibrium with Frank-Wolfe algorithm
- **Transit**: All-or-Nothing assignment on fixed networks
- Iterative equilibrium between demand and assignment

### Network
- 11 zones covering Stockholm metropolitan area
- Road network with capacity constraints and volume-delay functions
- Transit network (T-bana and tram) with in-vehicle and waiting time

## Data Files

- `zones.geojson` - Geographic zone boundaries
- `landuse.csv` - Population, employment, and income data
- `policy_comparison_results.csv` - Policy analysis results (generated)

## Key Features

✓ Multi-modal transportation model  
✓ Equilibrium assignment with convergence  
✓ Externalities calculation (noise, accidents, emissions, CO2)  
✓ Policy scenario analysis  
✓ Chinese and English documentation  

## Requirements

```
python >= 3.7
haversine
networkx
pandas
geopandas
numpy
tabulate
folium
```

## Citation

If you use this model in your research, please cite this repository.

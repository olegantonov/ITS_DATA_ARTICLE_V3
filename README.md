# The Impacts of ITS on Aviation Efficiency: PBN Implementation in Brazilian Air Routes

This repository contains the data processing scripts, statistical analysis pipeline, and visualization routines developed for the paper:

**Silva, D. G. M., Taco, P. W. G., & Arruda, F. S. (2025). _Impacts of ITS on Aviation Efficiency: PBN Implementation in Brazilian Air Routes_. University of Brasília – Graduate Program in Transport.**

The study examines the effects of **Performance-Based Navigation (PBN)**, an Intelligent Transportation System (ITS) technology, on the operational efficiency of Brazilian commercial aviation.  
Using ANAC microdata covering ≈544,000 flights on the 14 busiest domestic routes (2010–2024), we estimate the association between PBN adoption and **flight time, fuel consumption, and CO₂ emissions** through multiple linear regression models with robust errors and fixed effects.

---

## Repository structure

```
ITS_DATA_ARTICLE_V3/
├── scripts/
│   ├── its_pipeline_consolidated_v2.py     # Main data pipeline: filtering, PBN index construction, regression analysis
│   ├── analysis_visuals.py                 # Supplementary plots (PBN index evolution, regression coefficients, route comparisons)
├── results/
│   ├── tabela_regressao_tempo_voo_formatted.csv
│   ├── tabela_regressao_combustivel_formatted.csv
│   ├── tabela_regressao_co2_formatted.csv
│   ├── resumo_regressao_logTime_FE_rota.txt
│   └── ...
└── README.md                               # Project documentation (this file)
```

---

## Methodology summary

1. **Data source**: ANAC official microdata (2010–2024) for the 14 busiest domestic routes in Brazil.  
2. **Flight selection**:  
   - Status: *landed/arrived* (originally labeled `REALIZADO` in ANAC data).  
   - Aircraft: A320, A319, A321, B737, B738 (≥5% share).  
   - Routes: Filtered to top 14 O/D pairs.  
3. **PBN exposure index**:  
   - Constructed from documented airport-level implementation dates.  
   - Linear growth 0→1 between implementation date and 31 Dec 2024.  
   - Flight-level index = average of origin + destination values.  
   - Missing dates treated as `0` (conservative assumption).  
4. **Models**:  
   - OLS regressions with HC3 robust errors.  
   - Dependent variables: flight time, fuel burn, CO₂ emissions.  
   - Controls: route distance and aircraft dummies.  
   - Robustness: log-transformed flight time with route fixed effects.  
5. **CO₂ estimation**: ICAO conversion factor (1 ton fuel → 3.16 tons CO₂).  

---

## Key findings

- **Flight time**: Average reduction ≈ –2.6 minutes per flight (OLS).  
- **Fuel consumption**: Average reduction ≈ –27.7 kg per flight.  
- **CO₂ emissions**: Average reduction ≈ –87.6 kg per flight.  
- Aggregated across ≈544k flights, the system-level impact represents **tens of thousands of tons of avoided CO₂**.  

---

## Requirements

- Python ≥ 3.9  
- Dependencies:
  - pandas, numpy, statsmodels, scipy, tqdm, unidecode, matplotlib, seaborn, plotly  

---

## Usage

Outputs (figures + tables) will be stored in `./results`.

---

## Authors

- **Daniel Guilherme Marques da Silva** – University of Brasília, Graduate Program in Transport  
- **Pastor Willy Gonzales Taco** – University of Brasília, Graduate Program in Transport  
- **Fabiana Serra de Arruda** – University of Brasília, Graduate Program in Transport  

---

## License

This project is released under the MIT License.

---

## Citation

If you use this repository, please cite as:

```
Silva, D. G. M., Taco, P. W. G., & Arruda, F. S. (2025).
Impacts of ITS on Aviation Efficiency: PBN Implementation in Brazilian Air Routes.
University of Brasília – Graduate Program in Transport.
```

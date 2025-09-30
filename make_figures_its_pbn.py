

import os
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
BASE_PROJECT_PATH = r"C:\SCRIPTS\ITS_Article"
FILTERED_CSV_PATH = os.path.join(BASE_PROJECT_PATH, "unified_data_filtered_for_analysis.csv")
RESULTS_DIR       = os.path.join(BASE_PROJECT_PATH, "resultados_da_analise")
FIG_DIR           = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Arquivos de regressão (gerados pelo seu script principal)
TAB_TIME_MAIN   = os.path.join(RESULTS_DIR, "tabela_regressao_tempo_voo_formatted.csv")
TAB_FUEL_MAIN   = os.path.join(RESULTS_DIR, "tabela_regressao_combustivel_formatted.csv")
TAB_CO2_MAIN    = os.path.join(RESULTS_DIR, "tabela_regressao_co2_formatted.csv")

TAB_TIME_SENS   = os.path.join(RESULTS_DIR, "tabela_regressao_tempo_voo_sens_formatted.csv")
TAB_FUEL_SENS   = os.path.join(RESULTS_DIR, "tabela_regressao_combustivel_sens_formatted.csv")
TAB_CO2_SENS    = os.path.join(RESULTS_DIR, "tabela_regressao_co2_sens_formatted.csv")

CMP_MAIN_VS_SENS = os.path.join(RESULTS_DIR, "comparativo_coef_pbn_main_vs_sens.csv")

START_OF_WINDOW = date(2010, 1, 1)
END_OF_WINDOW   = date(2024, 12, 31)

# Cores/padronização leve (matplotlib puro)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# =========================================================


# ========================= UTILS =========================
def month_range(start_date: date, end_date: date):
    """Gera datas mensais (primeiro dia do mês) de start até end."""
    cur = date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

def linear_pbn_index_curve(start_impl: date | None,
                           start_window: date,
                           end_window: date,
                           round2: bool = True) -> pd.DataFrame:
    """Curva (mensal) do índice PBN: 0 antes; linear até 1 no end_window; 0 se None."""
    rows = []
    for m in month_range(start_window, end_window):
        if start_impl is None or m < start_impl:
            v = 0.0
        else:
            num = (min(m, end_window) - start_impl).days
            den = max((end_window - start_impl).days, 1)
            v = np.clip(num / den, 0.0, 1.0)
        rows.append((m, round(v, 2) if round2 else v))
    return pd.DataFrame(rows, columns=["month", "pbn_index"])

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def read_or_none(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None
# =========================================================


# ============ 1) FIGURA CONCEITUAL DO ÍNDICE PBN ============
def fig_concept_pbn_index():
    # Casos: Recife (2010-04-08), Brasília (2015-11-12), “sem data”
    r_start = date(2010, 4, 8)
    b_start = date(2015, 11, 12)
    n_start = None

    df_r = linear_pbn_index_curve(r_start, START_OF_WINDOW, END_OF_WINDOW)
    df_b = linear_pbn_index_curve(b_start, START_OF_WINDOW, END_OF_WINDOW)
    df_n = linear_pbn_index_curve(n_start, START_OF_WINDOW, END_OF_WINDOW)

    plt.figure()
    plt.plot(df_r["month"], df_r["pbn_index"], label="Airport A (start: 2010)")
    plt.plot(df_b["month"], df_b["pbn_index"], label="Airport B (start: 2015)")
    plt.plot(df_n["month"], df_n["pbn_index"], label="Airport C (undocumented)")
    plt.axvline(x=datetime(2010,1,1), ls="--", lw=1, color="gray")
    plt.axvline(x=datetime(2024,12,31), ls="--", lw=1, color="gray")
    plt.ylim(-0.05, 1.05)
    plt.title("Conceptual PBN Exposure Index (Monthly, 2010–2024)")
    plt.xlabel("Month")
    plt.ylabel("PBN index [0–1]")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "fig_concept_pbn_index.png"))


# =========== 2) DISTRIBUIÇÕES E HEXBIN A PARTIR DA BASE ===========
def load_filtered():
    if not os.path.exists(FILTERED_CSV_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {FILTERED_CSV_PATH}")
    df = pd.read_csv(FILTERED_CSV_PATH)
    # Expectativas mínimas de colunas (conforme seu script)
    need = ["tempo_voo_real_min","consumo_combustivel_kg","consumo_co2_do_voo_kg",
            "distancia_rota_nm","pbn_anos","velocidade_media_knots",
            "modelo","origem","destino","data_voo","rota_ordenada"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Coluna ausente no CSV filtrado: {c}")
    # Converter datas
    df["data_voo"] = pd.to_datetime(df["data_voo"])
    return df

def figs_distributions_and_hexbin(df: pd.DataFrame):
    # Histograma tempo de voo
    plt.figure()
    x = df["tempo_voo_real_min"].dropna()
    plt.hist(x, bins=50, alpha=0.8)
    plt.axvline(x.mean(), color="k", lw=1.5, ls="--", label=f"Mean = {x.mean():.1f} min")
    plt.axvline(x.median(), color="k", lw=1.0, ls=":", label=f"Median = {x.median():.1f} min")
    plt.title("Distribution of Actual Flight Time (min)")
    plt.xlabel("Minutes")
    plt.ylabel("Frequency")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "dist_tempo_voo.png"))

    # Histogramas Combustível e CO2
    for col, title, fname, unit in [
        ("consumo_combustivel_kg","Distribution of Fuel Consumption (kg)","dist_combustivel.png","kg"),
        ("consumo_co2_do_voo_kg","Distribution of CO₂ Emissions (kg)","dist_co2.png","kg"),
    ]:
        plt.figure()
        s = df[col].dropna()
        plt.hist(s, bins=50, alpha=0.8)
        plt.axvline(s.mean(), color="k", lw=1.5, ls="--", label=f"Mean = {s.mean():.0f} {unit}")
        plt.axvline(s.median(), color="k", lw=1.0, ls=":", label=f"Median = {s.median():.0f} {unit}")
        plt.title(title)
        plt.xlabel(unit)
        plt.ylabel("Frequency")
        plt.legend()
        savefig(os.path.join(FIG_DIR, fname))

    # Hexbin Distância vs Combustível / Tempo
    pairs = [
        ("distancia_rota_nm","consumo_combustivel_kg","Distance (NM)","Fuel (kg)","hexbin_dist_fuel.png"),
        ("distancia_rota_nm","tempo_voo_real_min","Distance (NM)","Time (min)","hexbin_dist_time.png")
    ]
    for xcol, ycol, xlabel, ylabel, fname in pairs:
        plt.figure()
        plt.hexbin(df[xcol], df[ycol], gridsize=40, mincnt=5, bins='log')
        plt.colorbar(label="log(count)")
        plt.title(f"{ylabel} vs. {xlabel}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        savefig(os.path.join(FIG_DIR, fname))


# =========== 3) BOXPLOTS POR MODELO E POR ROTA ===========
def figs_boxplots(df: pd.DataFrame):
    # Boxplots por modelo
    models = ["A320","A319","A321","B737","B738"]
    sub = df[df["modelo"].isin(models)].copy()
    for col, title, fname, ylab in [
        ("tempo_voo_real_min","Flight Time by Aircraft Model","box_time_by_model.png","Time (min)"),
        ("consumo_combustivel_kg","Fuel by Aircraft Model","box_fuel_by_model.png","Fuel (kg)"),
        ("consumo_co2_do_voo_kg","CO₂ by Aircraft Model","box_co2_by_model.png","CO₂ (kg)")
    ]:
        data = [sub.loc[sub["modelo"]==m, col].dropna() for m in models]
        plt.figure()
        plt.boxplot(data, labels=models, showfliers=False)
        plt.title(title)
        plt.ylabel(ylab)
        savefig(os.path.join(FIG_DIR, fname))

    # Boxplot de tempo por rota (ordenado por mediana)
    med = (df.groupby("rota_ordenada")["tempo_voo_real_min"]
             .median().sort_values())
    rotas_ord = med.index.tolist()
    data = [df.loc[df["rota_ordenada"]==r, "tempo_voo_real_min"].dropna() for r in rotas_ord]
    plt.figure(figsize=(12, 7))
    plt.boxplot(data, labels=rotas_ord, showfliers=False)
    plt.title("Flight Time by Route")
    plt.ylabel("Time (min)")
    plt.xticks(rotation=90)
    savefig(os.path.join(FIG_DIR, "box_time_by_route.png"))


# =========== 4) SÉRIE MENSAL: TEMPO MÉDIO + PBN MÉDIO ===========
def fig_monthly_time_and_pbn(df: pd.DataFrame):
    g = (df
         .set_index("data_voo")
         .resample("MS")
         .agg(time_mean=("tempo_voo_real_min","mean"),
              pbn_mean=("pbn_anos","mean"))
         .dropna())
    if g.empty:
        return
    fig, ax1 = plt.subplots()
    ax1.plot(g.index, g["time_mean"], lw=2, label="Flight time (mean)")
    ax1.set_ylabel("Time (min)")
    ax1.set_xlabel("Month")
    ax1.set_title("Monthly Flight Time (mean) and PBN Exposure (mean)")

    ax2 = ax1.twinx()
    ax2.plot(g.index, g["pbn_mean"], lw=2, ls="--", label="PBN index (mean)")
    ax2.set_ylabel("PBN index [0–1]")
    ax1.grid(True)
    # Legenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")
    savefig(os.path.join(FIG_DIR, "monthly_time_and_pbn.png"))


# =========== 5) GRÁFICOS DE COEFICIENTES (MAIN E SENS) ===========
def clean_reg_table(df_tab: pd.DataFrame) -> pd.DataFrame:
    # Espera colunas: Variable, Coefficient, Std. Error, t-stat, p-value
    # Mantém apenas as chaves de interesse e renomeia para labels curtos
    keep = ["PBN index","Route distance (NM)","A319","A321","B737","B738","Intercept"]
    df2 = df_tab[df_tab["Variable"].isin(keep)].copy()
    # Ordenação para plot
    order = ["PBN index","Route distance (NM)","A321","A319","B738","B737","Intercept"]
    df2["ord"] = df2["Variable"].apply(lambda v: order.index(v) if v in order else 999)
    df2 = df2.sort_values("ord")
    return df2

def coefplot_from_csv(tab_path: str, title: str, fname: str):
    if not os.path.exists(tab_path):
        return
    df_tab = pd.read_csv(tab_path)
    df2 = clean_reg_table(df_tab)
    if df2.empty:
        return
    ci = 1.96 * df2["Std. Error"].values
    beta = df2["Coefficient"].values
    x = np.arange(len(beta))
    labels = df2["Variable"].tolist()

    plt.figure(figsize=(10,6))
    plt.bar(x, beta, yerr=ci, capsize=4)
    plt.axhline(0, color="k", lw=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Coefficient (±95% CI)")
    savefig(os.path.join(FIG_DIR, fname))

def figs_coefplots():
    coefplot_from_csv(TAB_TIME_MAIN, "OLS (HC3) — Flight time (main)", "coef_time_main.png")
    coefplot_from_csv(TAB_FUEL_MAIN, "OLS (HC3) — Fuel (main)",        "coef_fuel_main.png")
    coefplot_from_csv(TAB_CO2_MAIN,  "OLS (HC3) — CO₂ (main)",         "coef_co2_main.png")

    # Sensibilidade
    coefplot_from_csv(TAB_TIME_SENS, "OLS (HC3) — Flight time (sensitivity)", "coef_time_sens.png")
    coefplot_from_csv(TAB_FUEL_SENS, "OLS (HC3) — Fuel (sensitivity)",        "coef_fuel_sens.png")
    coefplot_from_csv(TAB_CO2_SENS,  "OLS (HC3) — CO₂ (sensitivity)",         "coef_co2_sens.png")


# =========== 6) COMPARATIVO MAIN VS SENS (coef PBN) ===========
def fig_comp_main_vs_sens():
    if not os.path.exists(CMP_MAIN_VS_SENS):
        return
    dfc = pd.read_csv(CMP_MAIN_VS_SENS)
    # Espera colunas: Model, Main_beta, Main_p, Sens_beta, Sens_p
    # Plot simples de barras lado-a-lado por métrica
    plt.figure()
    idx = np.arange(len(dfc))
    width = 0.35
    plt.bar(idx - width/2, dfc["Main_beta"], width, label="Main")
    plt.bar(idx + width/2, dfc["Sens_beta"], width, label="Sensitivity")
    plt.axhline(0, color="k", lw=1)
    plt.xticks(idx, dfc["Model"].tolist(), rotation=20)
    plt.ylabel("β (PBN index)")
    plt.title("PBN coefficient: Main vs. Sensitivity")
    plt.legend()
    # Anotar p-values
    for i, (mb, mp, sb, sp) in enumerate(zip(dfc["Main_beta"], dfc["Main_p"], dfc["Sens_beta"], dfc["Sens_p"])):
        plt.text(i - width/2, mb, f"{mp}", ha="center", va="bottom", fontsize=8, rotation=90)
        plt.text(i + width/2, sb, f"{sp}", ha="center", va="bottom", fontsize=8, rotation=90)
    savefig(os.path.join(FIG_DIR, "comp_main_vs_sens_pbn.png"))


# ======================== MAIN RUNNER ========================
def main():
    print("[INFO] Gerando figura conceitual do índice PBN…")
    fig_concept_pbn_index()

    print("[INFO] Lendo base filtrada…")
    df = load_filtered()

    print("[INFO] Gerando distribuições e hexbin…")
    figs_distributions_and_hexbin(df)

    print("[INFO] Gerando boxplots por modelo e por rota…")
    figs_boxplots(df)

    print("[INFO] Gerando série mensal (tempo médio + PBN médio)…")
    fig_monthly_time_and_pbn(df)

    print("[INFO] Gerando coefplots (main e sens)…")
    figs_coefplots()

    print("[INFO] Gerando comparativo Main vs Sens (coef PBN)…")
    fig_comp_main_vs_sens()

    print(f"[OK] Figuras salvas em: {FIG_DIR}")

if __name__ == "__main__":
    main()

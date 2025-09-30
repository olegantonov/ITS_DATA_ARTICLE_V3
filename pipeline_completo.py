import os
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt
import logging
from datetime import datetime, date
from tqdm import tqdm

# ==============================================================================
# 1) CONFIGURAÇÃO GERAL E PARÂMETROS AJUSTÁVEIS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

BASE_PROJECT_PATH = r"C:\SCRIPTS\ITS_Article"
CLEANED_CSV_PATH  = os.path.join(BASE_PROJECT_PATH, "unified_data_cleaned.csv")
FILTERED_CSV_PATH = os.path.join(BASE_PROJECT_PATH, "unified_data_filtered_for_analysis.csv")
RESULTS_DIR       = os.path.join(BASE_PROJECT_PATH, "resultados_da_analise")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Filtros gerais (altere conforme taxonomia real da base)
ALLOWED_MODELS = {"B738", "A320", "A319", "B737", "A321"}
ALLOWED_CODIGO_TIPO_LINHA = {"N", "C"}  # ex.: N=Regular, C=Charter (ajuste se necessário)
ALLOWED_SITUACAO_VOO = "REALIZADO"       # equivalente a landed/arrived no paper
START_OF_WINDOW = date(2010, 1, 1)
END_OF_WINDOW   = date(2024, 12, 31)

# 14 rotas (pares brutos; para filtragem usamos chave ordenada)
ALLOWED_ROUTES_RAW = {
    ("SBSP","SBRJ"), ("SBSP","SBBR"), ("SBSP","SBCF"), ("SBPA","SBGR"),
    ("SBSV","SBGR"), ("SBRF","SBGR"), ("SBSP","SBPA"), ("SBSP","SBCT"),
    ("SBGR","SBCT"), ("SBGR","SBCF"), ("SBGR","SBBR"), ("SBRJ","SBBR"),
    ("SBGR","SBGL"), ("SBGR","SBFZ")
}
ALLOWED_ROUTES = {"-".join(sorted(pair)) for pair in ALLOWED_ROUTES_RAW}

# Distâncias **direcionais** (NM) — use para cálculo (mantém o sentido real do voo)
DIST_DICT = {
    ("SBSP", "SBRJ"): 197.44, ("SBRJ", "SBSP"): 197.44,
    ("SBSP", "SBBR"): 471.03, ("SBBR", "SBSP"): 471.03,
    ("SBSP", "SBCF"): 283.13, ("SBCF", "SBSP"): 283.13,
    ("SBPA", "SBGR"): 467.38, ("SBGR", "SBPA"): 467.38,
    ("SBSV", "SBGR"): 784.10, ("SBGR", "SBSV"): 784.10,
    ("SBRF", "SBGR"): 1134.38, ("SBGR", "SBRF"): 1134.38,
    ("SBSP", "SBPA"): 452.36, ("SBPA", "SBSP"): 452.36,
    ("SBSP", "SBCT"): 178.92, ("SBCT", "SBSP"): 178.92,
    ("SBGR", "SBCT"): 194.03, ("SBCT", "SBGR"): 194.03,
    ("SBGR", "SBCF"): 268.07, ("SBCF", "SBGR"): 268.07,
    ("SBGR", "SBBR"): 461.45, ("SBBR", "SBGR"): 461.45,
    ("SBRJ", "SBBR"): 501.03, ("SBBR", "SBRJ"): 501.03,
    ("SBGR", "SBGL"): 181.85, ("SBGL", "SBGR"): 181.85,
    ("SBGR", "SBFZ"): 1267.09, ("SBFZ", "SBGR"): 1267.09
}

# Tabelas teóricas de consumo por distância (kg)
CONSUMO_TABLES = {
    "A320": {125: 1672, 250: 3430, 500: 4585, 750: 6212, 1000: 7772, 1500: 10766},
    "A319": {125: 1596, 250: 3259, 500: 4323, 750: 5830, 1000: 7271, 1500: 10026},
    "B738": {125: 1715, 250: 3494, 500: 4621, 750: 6211, 1000: 7749, 1500: 10666},
    "B737": {125: 1695, 250: 3439, 500: 4515, 750: 6053, 1000: 7517, 1500: 10304},
    "A321": {125: 1909, 250: 3925, 500: 5270, 750: 7157, 1000: 8970, 1500: 12456}
}
CO2_FACTOR = 3.16  # CO2 total por voo = fuel_total * 3.16

# Datas exatas de implantação do PBN (DD/MM/YYYY); None = não implementado (undocumented)
IMPLANTATION_DATES = {
    'SBCF': '12/11/2015', 'SBBR': '12/11/2015', 'SBCT': '12/10/2017',
    'SBFZ': None,        'SBPA': '12/10/2017', 'SBRF': '08/04/2010',
    'SBRJ': '28/07/2011', 'SBGL': '28/07/2011', 'SBSV': '27/04/2017',
    'SBSP': '12/11/2015', 'SBGR': '12/11/2015'
}
IMPLANTATION_DATES = {
    k: (datetime.strptime(v, "%d/%m/%Y").date() if isinstance(v, str) and v.strip() else None)
    for k, v in IMPLANTATION_DATES.items()
}

# ===================== Parâmetros da remoção robusta de outliers =====================
Vmin_hard = 120.0   # knots 
Vmax_hard = 520.0   # knots 
MIN_GRP_STRICT = 200  # amostra mínima por rota p/ usar IQR+quantis p05/p95
MIN_GRP_RELAX  = 50   # se entre 50 e 199 obs, usar p10/p90 (+ hard caps)

# Overrides por rota (opcional): chave é a rota ordenada "AAA-BBB"
ROUTE_SPEED_BOUNDS = {
    # Ex.: "SBSP-SBCT": (110.0, 480.0)
}


USE_ROUTE_X_MODEL = False
# ==============================================================================

# =============================== Funções utilitárias ===========================
def safe_upper_strip(x: str) -> str:
    return str(x).strip().upper()

def ordered_route_key(orig: str, dest: str) -> str:
    return "-".join(sorted([safe_upper_strip(orig), safe_upper_strip(dest)]))

def compute_distance_nm(orig: str, dest: str) -> float:
    return DIST_DICT.get((safe_upper_strip(orig), safe_upper_strip(dest)))

def linear_pbn_index(flight_date: date, icao: str) -> float:
    """Índice PBN (0–1) para um aeroporto em uma data específica.
    - 0 antes da implantação ou se não implementado (None);
    - 0→1 linear do start até END_OF_WINDOW; 1 a partir de END_OF_WINDOW.

    [AJUSTE/EXPLICAÇÃO]: Quando a data de implantação é 'None' (undocumented),
    retornamos 0.0 em todas as datas. Essa convenção é conservadora e evita
    retro-datar exposição sem documentação verificável. O impacto dessa decisão
    é avaliado em análise de sensibilidade (ver adiante), excluindo tais voos.
    """
    start = IMPLANTATION_DATES.get(icao)
    if start is None:
        return 0.0
    if flight_date < start:
        return 0.0
    num = (min(flight_date, END_OF_WINDOW) - start).days
    den = max((END_OF_WINDOW - start).days, 1)
    return round(float(np.clip(num / den, 0.0, 1.0)), 2)

def pbn_index_for_flight(flight_date: date, orig: str, dest: str) -> float:
    i_o = linear_pbn_index(flight_date, safe_upper_strip(orig))
    i_d = linear_pbn_index(flight_date, safe_upper_strip(dest))
    return round((i_o + i_d) / 2.0, 2)

def has_undocumented_pbn(icao: str) -> bool:
    return IMPLANTATION_DATES.get(safe_upper_strip(icao)) is None

def interpolate_fuel(model: str, dist_nm: float) -> float:
    model = safe_upper_strip(model)
    if model not in CONSUMO_TABLES or pd.isna(dist_nm):
        return np.nan
    xs = sorted(CONSUMO_TABLES[model].keys())
    ys = [CONSUMO_TABLES[model][x] for x in xs]
    return float(np.interp(dist_nm, xs, ys))

# ======== Remoção de outliers por velocidade com limites robustos por rota ========
def robust_speed_bounds(group: pd.DataFrame):
    """Calcula (vmin, vmax) robustos para um grupo (rota ou rota×modelo)."""
    rota = group.name if isinstance(group.name, str) else group.name[0]
    if rota in ROUTE_SPEED_BOUNDS:
        vmin, vmax = ROUTE_SPEED_BOUNDS[rota]
        return max(vmin, Vmin_hard), min(vmax, Vmax_hard)

    v = group['velocidade_media_knots'].dropna().to_numpy()
    n = v.size
    if n >= MIN_GRP_STRICT:
        q1, q3 = np.quantile(v, [0.25, 0.75])
        iqr = q3 - q1
        p05, p95 = np.quantile(v, [0.05, 0.95])
        vmin = max(p05, q1 - 3.0*iqr, Vmin_hard)
        vmax = min(p95, q3 + 3.0*iqr, Vmax_hard)
    elif n >= MIN_GRP_RELAX:
        p10, p90 = np.quantile(v, [0.10, 0.90])
        vmin = max(p10, Vmin_hard)
        vmax = min(p90, Vmax_hard)
    else:
        vmin, vmax = Vmin_hard, Vmax_hard

    if vmin >= vmax:
        vmin, vmax = Vmin_hard, Vmax_hard
    return float(vmin), float(vmax)

# =============================== Pipeline principal ============================
def run_analysis_on_clean_data():
    logging.info("--- INICIANDO ANÁLISE (VERSÃO REVISADA COM OUTLIERS ROBUSTOS) ---")
    if not os.path.exists(CLEANED_CSV_PATH):
        logging.error(f"ARQUIVO LIMPO NÃO ENCONTRADO: {CLEANED_CSV_PATH}")
        return

    # Leitura em chunks
    total_lines = sum(1 for _ in open(CLEANED_CSV_PATH, 'r', encoding='utf-8'))
    chunk_size = 100000
    total_chunks = (total_lines // chunk_size) + 1

    df_chunks = []
    reader = pd.read_csv(CLEANED_CSV_PATH, sep=',', low_memory=False, chunksize=chunk_size, on_bad_lines='skip')
    for chunk in tqdm(reader, total=total_chunks, desc="Lendo arquivo limpo"):
        df_chunks.append(chunk)
    df = pd.concat(df_chunks, ignore_index=True)
    logging.info("Arquivo limpo carregado com sucesso.")

    # ---------------------- Normalização de colunas ---------------------------
    rename_map = {
        'situaaaaao_voo': 'situacao_voo',
        'caa3digo_tipo_linha': 'codigo_tipo_linha',
        'sigla_icao_aeroporto_origem': 'origem',
        'sigla_icao_aeroporto_destino': 'destino',
        'modelo_equipamento': 'modelo',
        'partida_real': 'partida_real',
        'chegada_real': 'chegada_real'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    required = ['situacao_voo','codigo_tipo_linha','origem','destino','modelo','partida_real','chegada_real']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória ausente após normalização: {col}")

    # ---------------------- Filtragem base ------------------------------------
    df.dropna(subset=['partida_real','chegada_real','origem','destino'], inplace=True)
    df['origem']  = df['origem'].astype(str).str.strip().str.upper()
    df['destino'] = df['destino'].astype(str).str.strip().str.upper()

    df['partida_real'] = pd.to_datetime(df['partida_real'], errors='coerce')
    df['chegada_real'] = pd.to_datetime(df['chegada_real'], errors='coerce')
    df.dropna(subset=['partida_real','chegada_real'], inplace=True)

    df['data_voo'] = df['partida_real'].dt.date
    df = df[(df['data_voo'] >= START_OF_WINDOW) & (df['data_voo'] <= END_OF_WINDOW)].copy()

    df['rota_ordenada'] = df.apply(lambda r: ordered_route_key(r['origem'], r['destino']), axis=1)

    mask = (
        (df['situacao_voo'].astype(str).str.strip().str.upper() == ALLOWED_SITUACAO_VOO) &
        (df['codigo_tipo_linha'].astype(str).str.strip().str.upper().isin(ALLOWED_CODIGO_TIPO_LINHA)) &
        (df['modelo'].astype(str).str.strip().str.upper().isin(ALLOWED_MODELS)) &
        (df['rota_ordenada'].isin(ALLOWED_ROUTES))
    )
    df = df[mask].copy()
    logging.info(f"Filtragem concluída. Total de {len(df)} registros elegíveis.")

    # ---------------------- Engenharia de variáveis (pré-outliers) ------------
    df['distancia_rota_nm']   = df.apply(lambda r: compute_distance_nm(r['origem'], r['destino']), axis=1)
    df['tempo_voo_real_min']  = (df['chegada_real'] - df['partida_real']).dt.total_seconds() / 60.0
    df.dropna(subset=['distancia_rota_nm','tempo_voo_real_min'], inplace=True)
    df = df[df['tempo_voo_real_min'] > 0].copy()

    # Velocidade média (knots)
    df['velocidade_media_knots'] = df['distancia_rota_nm'] / (df['tempo_voo_real_min'] / 60.0)

    # ---------------------- Remoção robusta de outliers por rota -------------
    group_cols = ['rota_ordenada'] if not USE_ROUTE_X_MODEL else ['rota_ordenada','modelo']

    bounds = (
        df.groupby(group_cols, observed=True)
          .apply(robust_speed_bounds)
          .rename('bounds')
          .reset_index()
    )
    bounds[['vmin','vmax']] = pd.DataFrame(bounds['bounds'].tolist(), index=bounds.index)
    bounds.drop(columns=['bounds'], inplace=True)

    df = df.merge(bounds, on=group_cols, how='left')

    before = len(df)
    df = df[(df['velocidade_media_knots'] >= df['vmin']) & (df['velocidade_media_knots'] <= df['vmax'])].copy()
    removed = before - len(df)
    logging.info(f"Remoção robusta concluída. Removidos: {removed} ({removed/max(before,1):.2%}). Restantes: {len(df)}.")

    # Diagnóstico de limites por rota
    diag = (
        df.groupby('rota_ordenada', observed=True)
          .agg(n=('velocidade_media_knots','size'),
               vmin=('vmin','first'), vmax=('vmax','first'),
               v_mean=('velocidade_media_knots','mean'), v_std=('velocidade_media_knots','std'))
          .reset_index()
    )
    diag.to_csv(os.path.join(RESULTS_DIR, "diagnostico_limites_velocidade_por_rota.csv"), index=False, encoding='utf-8')

    # ---------------------- Engenharia de variáveis (final) -------------------
    df['consumo_combustivel_kg'] = df.apply(lambda r: interpolate_fuel(r['modelo'], r['distancia_rota_nm']), axis=1)
    df['consumo_co2_do_voo_kg']  = df['consumo_combustivel_kg'] * CO2_FACTOR
    df['pbn_anos'] = df.apply(lambda r: pbn_index_for_flight(r['data_voo'], r['origem'], r['destino']), axis=1)

    # [AJUSTE] Flag para amostra de sensibilidade (origem OU destino sem data PBN documentada)
    df['od_has_undoc'] = df.apply(
        lambda r: has_undocumented_pbn(r['origem']) or has_undocumented_pbn(r['destino']),
        axis=1
    )

    df.dropna(subset=['consumo_combustivel_kg','consumo_co2_do_voo_kg'], inplace=True)

    # Salva base filtrada para reprodutibilidade
    df.to_csv(FILTERED_CSV_PATH, index=False, encoding='utf-8')
    logging.info(f"Base filtrada salva em: {FILTERED_CSV_PATH}")

    # ---------------------- Estatísticas descritivas ---------------------------
    stats_cols = ['tempo_voo_real_min','consumo_combustivel_kg','consumo_co2_do_voo_kg',
                  'distancia_rota_nm','pbn_anos','velocidade_media_knots']
    stats_df = df[df['tempo_voo_real_min'] <= 500][stats_cols]
    descricao = stats_df.describe().loc[['mean','std','min','max']].T
    descricao.columns = ['Média','Desvio Padrão','Mínimo','Máximo']
    descricao.to_csv(os.path.join(RESULTS_DIR, "tabela_estatisticas_descritivas.csv"), float_format='%.4f')

    # ====================== REGRESSÕES + P-VALUES FORMATADOS ===================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Força referência A320 e restringe às 5 categorias de interesse
    cats = ['A320','A321','A319','B737','B738']  # ordem apenas para garantir presença
    df = df[df['modelo'].isin(cats)].copy()
    df['modelo'] = pd.Categorical(df['modelo'], categories=cats, ordered=True)

    # Utilitário: formato "p = 1×10⁻ᵏ" com expoente sobrescrito; decimal para p>=0,001
    _SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    def format_p_sci1(p: float) -> str:
        if p is None or pd.isna(p):
            return "p = n/a"
        if p < 1e-300:
            return "p < 1×10" + "⁻³⁰⁰"
        if p >= 1e-3:
            return f"p = {p:.3f}"
        exp = int(math.floor(math.log10(p)))
        return "p = 1×10" + str(exp).translate(_SUP)

    NICE = {
        "Intercept": "Intercept",
        "pbn_anos": "PBN index",
        "distancia_rota_nm": "Route distance (NM)",
        'C(modelo, Treatment(reference="A320"))[T.A319]': "A319",
        'C(modelo, Treatment(reference="A320"))[T.A321]': "A321",
        'C(modelo, Treatment(reference="A320"))[T.B737]': "B737",
        'C(modelo, Treatment(reference="A320"))[T.B738]': "B738",
    }

    def run_ols_and_export(dep_col: str, file_stub: str):
        formula = (
            f'{dep_col} ~ pbn_anos + distancia_rota_nm '
            f'+ C(modelo, Treatment(reference="A320"))'
        )
        model = smf.ols(formula, data=df).fit(cov_type='HC3')

        out = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.round(3),
            "Std. Error": model.bse.round(3),
            "t-stat": model.tvalues.round(2),
            "p-value": [format_p_sci1(p) for p in model.pvalues]
        })
        out["Variable"] = out["Variable"].map(lambda v: NICE.get(v, v))

        meta = pd.DataFrame({
            "Metric": ["N", "R-squared", "Adj. R-squared"],
            "Value": [int(model.nobs), round(model.rsquared, 3), round(model.rsquared_adj, 3)]
        })

        csv_path = os.path.join(RESULTS_DIR, f"tabela_regressao_{file_stub}_formatted.csv")
        meta_path = os.path.join(RESULTS_DIR, f"metricas_{file_stub}.csv")
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

        # (Opcional) também salvar o summary cru para auditoria
        with open(os.path.join(RESULTS_DIR, f"resumo_regressao_{file_stub}.txt"), "w", encoding="utf-8") as f:
            f.write(str(model.summary()))

        logging.info(f"[OK] {file_stub}: Tabela e métricas salvas.")
        return model, out, meta

    # Executa para as três dependentes (amostra principal)
    m_time, tab_time, meta_time = run_ols_and_export("tempo_voo_real_min", "tempo_voo")
    m_fuel, tab_fuel, meta_fuel = run_ols_and_export("consumo_combustivel_kg", "combustivel")
    m_co2,  tab_co2,  meta_co2  = run_ols_and_export("consumo_co2_do_voo_kg", "co2")

    # ---------------------- Variante: log(Time) + FE por rota ------------------
    try:
        df_fe = df[df["tempo_voo_real_min"] > 0].copy()
        df_fe["route_id"] = df_fe["origem"].str.cat(df_fe["destino"], sep='-')
        formula_fe = (
            'np.log(tempo_voo_real_min) ~ pbn_anos + distancia_rota_nm '
            '+ C(route_id) + C(modelo, Treatment(reference="A320"))'
        )
        m_log_fe = smf.ols(formula_fe, data=df_fe).fit(
            cov_type='cluster', cov_kwds={'groups': df_fe['route_id']}
        )
        p_pbn = m_log_fe.pvalues.get('pbn_anos', float('nan'))
        resumo_fe = pd.DataFrame({
            "Metric": ["N", "R-squared", "PBN p-value (formatted)"],
            "Value": [int(m_log_fe.nobs), round(m_log_fe.rsquared, 3), format_p_sci1(p_pbn)]
        })
        resumo_fe.to_csv(os.path.join(RESULTS_DIR, "resumo_logTime_FE_rota.csv"),
                         index=False, encoding="utf-8-sig")
        with open(os.path.join(RESULTS_DIR, "resumo_regressao_logTime_FE_rota.txt"), "w", encoding="utf-8") as f:
            f.write(str(m_log_fe.summary()))
        logging.info("[OK] Modelo log(Time)+FE por rota salvo.")
    except Exception as e:
        logging.warning(f"Falha ao estimar variante log(Time)+FE por rota: {e}")

    # ======================= ANÁLISE DE SENSIBILIDADE ============================
    df_sens = df[~df['od_has_undoc']].copy()
    logging.info(f"[Sens] Registros após excluir O/D com PBN undocumented: {len(df_sens)}")

    def run_ols_and_export_on(df_in: pd.DataFrame, dep_col: str, file_stub: str):
        formula = (
            f'{dep_col} ~ pbn_anos + distancia_rota_nm '
            f'+ C(modelo, Treatment(reference="A320"))'
        )
        model = smf.ols(formula, data=df_in).fit(cov_type='HC3')
        out = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.round(3),
            "Std. Error": model.bse.round(3),
            "t-stat": model.tvalues.round(2),
            "p-value": [format_p_sci1(p) for p in model.pvalues]
        })
        out["Variable"] = out["Variable"].map(lambda v: NICE.get(v, v))
        meta = pd.DataFrame({
            "Metric": ["N", "R-squared", "Adj. R-squared"],
            "Value": [int(model.nobs), round(model.rsquared, 3), round(model.rsquared_adj, 3)]
        })
        out.to_csv(os.path.join(RESULTS_DIR, f"tabela_regressao_{file_stub}_formatted.csv"),
                   index=False, encoding="utf-8-sig")
        meta.to_csv(os.path.join(RESULTS_DIR, f"metricas_{file_stub}.csv"),
                    index=False, encoding="utf-8-sig")
        with open(os.path.join(RESULTS_DIR, f"resumo_regressao_{file_stub}.txt"), "w", encoding="utf-8") as f:
            f.write(str(model.summary()))
        logging.info(f"[Sens-OK] {file_stub}: Tabela e métricas salvas.")
        return model, out, meta

    # Executa as três dependentes na amostra de sensibilidade
    if len(df_sens) > 0:
        m_time_s, tab_time_s, meta_time_s = run_ols_and_export_on(df_sens, "tempo_voo_real_min", "tempo_voo_sens")
        m_fuel_s, tab_fuel_s, meta_fuel_s = run_ols_and_export_on(df_sens, "consumo_combustivel_kg", "combustivel_sens")
        m_co2_s,  tab_co2_s,  meta_co2_s  = run_ols_and_export_on(df_sens, "consumo_co2_do_voo_kg", "co2_sens")

        # --- Comparativo rápido: coeficiente de pbn_anos no modelo principal vs. sensibilidade
        def coef_row(label, model_main, model_sens, var="pbn_anos"):
            return {
                "Model": label,
                "Main_beta": round(float(model_main.params.get(var, np.nan)), 4),
                "Main_p": format_p_sci1(float(model_main.pvalues.get(var, np.nan))),
                "Sens_beta": round(float(model_sens.params.get(var, np.nan)), 4),
                "Sens_p": format_p_sci1(float(model_sens.pvalues.get(var, np.nan)))
            }

        cmp = pd.DataFrame([
            coef_row("Flight time (min)", m_time, m_time_s),
            coef_row("Fuel (kg)",        m_fuel, m_fuel_s),
            coef_row("CO2 (kg)",         m_co2,  m_co2_s),
        ])
        cmp.to_csv(os.path.join(RESULTS_DIR, "comparativo_coef_pbn_main_vs_sens.csv"),
                   index=False, encoding="utf-8-sig")
        logging.info("[Sens] Comparativo de coeficientes salvo: comparativo_coef_pbn_main_vs_sens.csv")
    else:
        logging.warning("[Sens] Amostra de sensibilidade vazia após filtro de O/D com PBN undocumented.")

    logging.info("--- ANÁLISE CONCLUÍDA ---")

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    logging.info(">>> INICIANDO SCRIPT DE ANÁLISE (OUTLIERS ROBUSTOS POR ROTA) <<<")
    run_analysis_on_clean_data()
    logging.info(">>> SCRIPT CONCLUÍDO <<<")

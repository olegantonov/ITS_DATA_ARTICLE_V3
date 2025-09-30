
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURAÇÃO GERAL
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

BASE_PROJECT_PATH = r"C:\SCRIPTS\ITS_Article"
CLEANED_CSV_PATH = os.path.join(BASE_PROJECT_PATH, "unified_data_cleaned.csv") 
FILTERED_CSV_PATH = os.path.join(BASE_PROJECT_PATH, "unified_data_filtered_for_analysis.csv")
RESULTS_DIR = os.path.join(BASE_PROJECT_PATH, "resultados_da_analise")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.info(f"Diretórios de trabalho configurados.")
logging.info(f"Lendo arquivo de dados limpo: {CLEANED_CSV_PATH}")

# ==============================================================================
# 2. FUNÇÃO DE ANÁLISE (COM ROBUSTEZ CIENTÍFICA)
# ==============================================================================
def run_analysis_on_clean_data():
    logging.info("--- INICIANDO ANÁLISE DO ARQUIVO LIMPO ---")
    if not os.path.exists(CLEANED_CSV_PATH):
        logging.error(f"ARQUIVO LIMPO NÃO ENCONTRADO: {CLEANED_CSV_PATH}.")
        return

    # Parâmetros de Análise
    ALLOWED_MODELS = {"B738", "A320", "A319", "B737", "A321"}
    ALLOWED_CODIGO_TIPO_LINHA = {"N", "C"}
    ALLOWED_SITUACAO_VOO = "REALIZADO"
    ALLOWED_ROUTES = { "SBRJ-SBSP", "SBBR-SBSP", "SBCF-SBSP", "SBPA-SBGR", "SBGR-SBSV", "SBGR-SBRF", "SBPA-SBSP", "SBCT-SBSP", "SBCT-SBGR", "SBCF-SBGR", "SBBR-SBGR", "SBBR-SBRJ", "SBGL-SBGR", "SBFZ-SBGR" }
    DIST_DICT = { ("SBSP", "SBRJ"): 197.44, ("SBRJ", "SBSP"): 197.44, ("SBSP", "SBBR"): 471.03, ("SBBR", "SBSP"): 471.03, ("SBSP", "SBCF"): 283.13, ("SBCF", "SBSP"): 283.13, ("SBPA", "SBGR"): 467.38, ("SBGR", "SBPA"): 467.38, ("SBSV", "SBGR"): 784.10, ("SBGR", "SBSV"): 784.10, ("SBRF", "SBGR"): 1134.38, ("SBGR", "SBRF"): 1134.38, ("SBSP", "SBPA"): 452.36, ("SBPA", "SBSP"): 452.36, ("SBSP", "SBCT"): 178.92, ("SBCT", "SBSP"): 178.92, ("SBGR", "SBCT"): 194.03, ("SBCT", "SBGR"): 194.03, ("SBGR", "SBCF"): 268.07, ("SBCF", "SBGR"): 268.07, ("SBGR", "SBBR"): 461.45, ("SBBR", "SBGR"): 461.45, ("SBRJ", "SBBR"): 501.03, ("SBBR", "SBRJ"): 501.03, ("SBGR", "SBGL"): 181.85, ("SBGL", "SBGR"): 181.85, ("SBGR", "SBFZ"): 1267.09, ("SBFZ", "SBGR"): 1267.09 }
    CONSUMO_TABLES = { "A320": {125: 1672, 250: 3430, 500: 4585, 750: 6212, 1000: 7772, 1500: 10766}, "A319": {125: 1596, 250: 3259, 500: 4323, 750: 5830, 1000: 7271, 1500: 10026}, "B738": {125: 1715, 250: 3494, 500: 4621, 750: 6211, 1000: 7749, 1500: 10666}, "B737": {125: 1695, 250: 3439, 500: 4515, 750: 6053, 1000: 7517, 1500: 10304}, "A321": {125: 1909, 250: 3925, 500: 5270, 750: 7157, 1000: 8970, 1500: 12456} }
    CO2_FACTOR = 3.16
    PBN_IMPLANTATION_YEARS = { 'SBCF': 2015, 'SBBR': 2015, 'SBCT': 2017, 'SBFZ': 0, 'SBPA': 2017, 'SBRF': 2010, 'SBRJ': 2011, 'SBGL': 2011, 'SBSV': 2017, 'SBSP': 2015, 'SBGR': 2015 }

    try:
        # Lendo o arquivo limpo em chunks
        total_lines = sum(1 for _ in open(CLEANED_CSV_PATH, 'r', encoding='utf-8'))
        chunk_size = 100000
        total_chunks = (total_lines // chunk_size) + 1
        
        df_chunks = []
        chunk_iterator = pd.read_csv(CLEANED_CSV_PATH, sep=',', low_memory=False, chunksize=chunk_size, on_bad_lines='skip')
        for chunk in tqdm(chunk_iterator, total=total_chunks, desc="Lendo arquivo limpo"):
            df_chunks.append(chunk)
        df = pd.concat(df_chunks, ignore_index=True)
        logging.info("Arquivo limpo carregado com sucesso.")

        # Filtragem inicial
        df.dropna(subset=['partida_real', 'chegada_real', 'sigla_icao_aeroporto_origem', 'sigla_icao_aeroporto_destino'], inplace=True)
        df['rota_ordenada'] = df.apply(lambda row: "-".join(sorted([row['sigla_icao_aeroporto_origem'].strip().upper(), row['sigla_icao_aeroporto_destino'].strip().upper()])), axis=1)
        
        mask = (
            (df['situaaaaao_voo'].astype(str).str.strip().str.upper() == ALLOWED_SITUACAO_VOO) &
            (df['caa3digo_tipo_linha'].astype(str).str.strip().str.upper().isin(ALLOWED_CODIGO_TIPO_LINHA)) &
            (df['modelo_equipamento'].astype(str).str.strip().str.upper().isin(ALLOWED_MODELS)) &
            (df['rota_ordenada'].isin(ALLOWED_ROUTES))
        )
        df = df[mask].copy()
        logging.info(f"Filtragem inicial concluída. Total de {len(df)} registros válidos.")

        # Engenharia de Variáveis preliminar para remoção de outliers
        df['partida_real'] = pd.to_datetime(df['partida_real'], errors='coerce')
        df['chegada_real'] = pd.to_datetime(df['chegada_real'], errors='coerce')
        df['distancia_rota_nm'] = df.apply(lambda r: DIST_DICT.get((r['sigla_icao_aeroporto_origem'].strip(), r['sigla_icao_aeroporto_destino'].strip())), axis=1)
        df['tempo_voo_real_min'] = (df['chegada_real'] - df['partida_real']).dt.total_seconds() / 60
        df.dropna(subset=['distancia_rota_nm', 'tempo_voo_real_min'], inplace=True)
        df = df[df['tempo_voo_real_min'] > 0].copy()

        # --- NOVO: Remoção Científica de Outliers (IQR por Rota) ---
        logging.info("Iniciando remoção robusta de outliers (método IQR por rota)...")
        df['velocidade_media_knots'] = df['distancia_rota_nm'] / (df['tempo_voo_real_min'] / 60)
        
        # Calcula os limites IQR para cada rota
        q1 = df.groupby('rota_ordenada')['velocidade_media_knots'].transform('quantile', 0.25)
        q3 = df.groupby('rota_ordenada')['velocidade_media_knots'].transform('quantile', 0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filtra o DataFrame, mantendo apenas os voos dentro dos limites de sua respectiva rota
        initial_rows = len(df)
        df = df[(df['velocidade_media_knots'] >= lower_bound) & (df['velocidade_media_knots'] <= upper_bound)]
        outliers_removed = initial_rows - len(df)
        logging.info(f"Remoção de outliers concluída. {outliers_removed} registros removidos.")
        logging.info(f"Total de registros para análise robusta: {len(df)}.")

        # Salvando o DataFrame final que entra na análise
        logging.info(f"Salvando os {len(df)} registros finais para '{FILTERED_CSV_PATH}'...")
        df.to_csv(FILTERED_CSV_PATH, index=False, encoding='utf-8')
        logging.info("Arquivo final salvo com sucesso.")
        
        # Engenharia de Variáveis final
        logging.info("Iniciando engenharia de variáveis final...")
        df['ano'] = df['partida_real'].dt.year
        df['consumo_combustivel_kg'] = df.apply(lambda r: np.interp(r['distancia_rota_nm'], list(CONSUMO_TABLES[r['modelo_equipamento']].keys()), list(CONSUMO_TABLES[r['modelo_equipamento']].values())) if r['modelo_equipamento'] in CONSUMO_TABLES and pd.notna(r['distancia_rota_nm']) else np.nan, axis=1)
        df['consumo_co2_do_voo_kg'] = df['consumo_combustivel_kg'] * CO2_FACTOR
        
        df.dropna(subset=['ano', 'consumo_combustivel_kg'], inplace=True)
        logging.info(f"Engenharia de variáveis concluída. Total final: {len(df)}.")

        def compute_pbn_index(row):
            ano_voo, orig, dest = row['ano'], row['sigla_icao_aeroporto_origem'], row['sigla_icao_aeroporto_destino']
            get_index = lambda icao: round(min(1.0, (ano_voo - PBN_IMPLANTATION_YEARS.get(icao, 0) + 1) / (2024 - PBN_IMPLANTATION_YEARS.get(icao, 0) + 1)), 2) if ano_voo >= PBN_IMPLANTATION_YEARS.get(icao, 0) and PBN_IMPLANTATION_YEARS.get(icao, 0) != 0 else 0
            return round((get_index(orig) + get_index(dest)) / 2, 2)
        df['pbn_anos'] = df.apply(compute_pbn_index, axis=1)
        
        # Estatísticas Descritivas
        logging.info("Gerando estatísticas descritivas...")
        stats_df = df[df['tempo_voo_real_min'] <= 500][['tempo_voo_real_min', 'consumo_combustivel_kg', 'consumo_co2_do_voo_kg', 'distancia_rota_nm', 'pbn_anos']]
        descricao = stats_df.describe().loc[['mean', 'std', 'min', 'max']].T
        descricao.columns = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
        descricao.to_csv(os.path.join(RESULTS_DIR, "tabela_estatisticas_descritivas.csv"), float_format='%.4f')
        print("\n--- Tabela de Estatísticas Descritivas ---")
        print(descricao.to_string(float_format='%.4f'))
        
        # Análise de Regressão com Diagnósticos Completos
        from statsmodels.stats.diagnostic import het_breuschpagan
        import scipy.stats as st

        logging.info("Iniciando análise de regressão com diagnósticos completos...")
        df_modelo = pd.get_dummies(df, columns=['modelo_equipamento'], drop_first=True)
        indep_vars = ['pbn_anos', 'distancia_rota_nm']
        dummy_cols = [c for c in df_modelo.columns if c.startswith('modelo_equipamento_')]
        X = sm.add_constant(df_modelo[indep_vars + dummy_cols]).astype(float)
        dependent_vars = {"Tempo de Voo Real (min)": df_modelo['tempo_voo_real_min'], "Consumo de Combustível (kg)": df_modelo['consumo_combustivel_kg'],"Emissão de CO2 (kg)": df_modelo['consumo_co2_do_voo_kg']}
        
        for name, y_var in dependent_vars.items():
            logging.info(f"Ajustando modelo e diagnósticos para: {name}...")
            modelo = sm.OLS(y_var.astype(float), X).fit(cov_type='HC3')
            
            with open(os.path.join(RESULTS_DIR, f"resumo_regressao_{name.replace(' ', '_')}.txt"), 'w', encoding='utf-8') as f:
                f.write(str(modelo.summary()))
                f.write("\n\n" + "="*80)
                f.write("\n--- TESTES DE DIAGNÓSTICO ---\n")
                f.write("="*80 + "\n")

                bp_test = het_breuschpagan(modelo.resid, modelo.model.exog)
                f.write("\n1. Teste de Breusch-Pagan (Heterocedasticidade):\n")
                f.write(f"   - Estatística LM: {bp_test[0]:.4f}\n")
                f.write(f"   - p-valor do teste LM: {bp_test[1]:.4f}\n")
                f.write(f"   - Conclusão: {'Heterocedasticidade detectada (p < 0.05).' if bp_test[1] < 0.05 else 'Não há evidência de heterocedasticidade (p >= 0.05).'}\n")

                stat, p_valor = st.normaltest(modelo.resid)
                f.write(f"\n2. Teste de D'Agostino & Pearson (Normalidade dos Resíduos):\n")
                f.write(f"   - Estatística: {stat:.4f}\n")
                f.write(f"   - p-valor: {p_valor:.6f}\n")
                f.write(f"   - Conclusão: {'Resíduos NÃO seguem uma distribuição normal (p < 0.05).' if p_valor < 0.05 else 'Não há evidências para rejeitar a normalidade dos resíduos (p >= 0.05).'}\n")

        logging.info("Modelos de regressão e diagnósticos foram ajustados e salvos.")
    
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante a análise: {e}", exc_info=True)

# ==============================================================================
# ORQUESTRADOR PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    logging.info(">>> INICIANDO SCRIPT DE ANÁLISE ROBUSTA (A PARTIR DO ARQUIVO LIMPO) <<<")
    
    run_analysis_on_clean_data()
    
    logging.info(">>> SCRIPT CONCLUÍDO <<<")
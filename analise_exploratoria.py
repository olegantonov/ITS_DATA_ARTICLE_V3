import pandas as pd
import os
import unicodedata
import re
import logging

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# --- Caminho para o seu arquivo de dados ---
FILE_PATH = r"C:\SCRIPTS\ITS_Article\unified_data_cleaned.csv"
CLEANED_FILE_PATH = r"C:\SCRIPTS\ITS_Article\unified_data_cleaned.csv"

def analyze_and_clean_csv(file_path, cleaned_path):
    """
    Executa uma análise completa do arquivo CSV e demonstra como limpá-lo.
    """
    if not os.path.exists(file_path):
        logging.error(f"ARQUIVO NÃO ENCONTRADO: {file_path}")
        return

    # --- ETAPA 1: DETECÇÃO AUTOMÁTICA DE FORMATO ---
    logging.info("Iniciando detecção automática de formato (separador e encoding)...")
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            first_line = f.readline()
        
        # Detecta o separador
        sep = ';' if first_line.count(';') > first_line.count(',') else ','
        encoding = 'latin-1' # Assumindo latin-1 com base nos erros anteriores
        
        logging.info(f"Formato detectado: Separador='{sep}', Encoding='{encoding}'")
        
        # Carrega uma pequena amostra para análise
        df_sample = pd.read_csv(file_path, sep=sep, encoding=encoding, on_bad_lines='skip', nrows=1000)

    except Exception as e:
        logging.error(f"Falha ao ler o arquivo. Verifique se o caminho está correto. Erro: {e}")
        return

    # --- ETAPA 2: ANÁLISE DA ESTRUTURA E CONTEÚDO ---
    print("\n" + "="*80)
    print("ETAPA 2.1: AMOSTRA DOS DADOS (5 PRIMEIRAS LINHAS)")
    print("="*80)
    print(df_sample.head().to_string())
    
    print("\n" + "="*80)
    print("ETAPA 2.2: INFORMAÇÕES GERAIS DAS COLUNAS (TIPOS DE DADOS E VALORES NULOS)")
    print("="*80)
    df_sample.info()

    print("\n" + "="*80)
    print("ETAPA 2.3: ANÁLISE DE VALORES ÚNICOS EM COLUNAS CATEGÓRICAS IMPORTANTES")
    print("="*80)
    categorical_cols_to_check = [
        'Situação Voo', 'Código Tipo Linha', 'Modelo Equipamento', 'Situação Partida', 'Situação Chegada'
    ]
    for col in categorical_cols_to_check:
        if col in df_sample.columns:
            print(f"\n--- Valores Únicos para a coluna: '{col}' ---")
            # Remove valores nulos (NaN) antes de contar
            unique_values = df_sample[col].dropna().unique()
            print(unique_values)
        else:
            print(f"\nAVISO: A coluna '{col}' não foi encontrada na amostra.")

    # --- ETAPA 3: NORMALIZAÇÃO E LIMPEZA ---
    def normalize_column_names(columns):
        """Função para limpar e padronizar os nomes das colunas."""
        normalized_columns = []
        for col in columns:
            col_str = str(col).lower()
            nfkd_form = unicodedata.normalize('NFKD', col_str)
            ascii_col = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
            ascii_col = re.sub(r'[^a-z0-9]+', '_', ascii_col).strip('_')
            normalized_columns.append(ascii_col)
        return normalized_columns

    print("\n" + "="*80)
    print("ETAPA 3.1: SUGESTÃO DE NORMALIZAÇÃO DOS NOMES DAS COLUNAS")
    print("="*80)
    original_columns = df_sample.columns
    normalized_columns = normalize_column_names(original_columns)
    mapping = {orig: norm for orig, norm in zip(original_columns, normalized_columns)}
    for orig, norm in mapping.items():
        print(f'"{orig}"  ->  "{norm}"')
        
    print("\n" + "="*80)
    print("ETAPA 3.2: COMO CRIAR UMA CÓPIA LIMPA DO ARQUIVO")
    print("="*80)
    print(f"O próximo passo seria ler o arquivo completo, aplicar a normalização e salvar em '{cleaned_path}'.")
    print("Para fazer isso, você pode descomentar o bloco de código 'CRIAR CÓPIA LIMPA' neste script e executá-lo novamente.")

    # --- (OPCIONAL) BLOCO DE CÓDIGO PARA CRIAR A CÓPIA LIMPA ---
    # Para executar a limpeza, remova os três apóstrofos (''') do início e do fim deste bloco.
    '''
    logging.info("Iniciando a criação do arquivo limpo. Isso pode demorar alguns minutos...")
    try:
        chunk_iterator = pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False, chunksize=100000, on_bad_lines='skip')
        
        is_first = True
        for chunk in chunk_iterator:
            chunk.columns = normalize_column_names(chunk.columns)
            
            if is_first:
                chunk.to_csv(cleaned_path, mode='w', index=False, encoding='utf-8')
                is_first = False
            else:
                chunk.to_csv(cleaned_path, mode='a', header=False, index=False, encoding='utf-8')
        
        logging.info(f"SUCESSO! Arquivo limpo foi criado em: {cleaned_path}")

    except Exception as e:
        logging.error(f"Falha ao criar o arquivo limpo. Erro: {e}")
    '''

if __name__ == "__main__":
    analyze_and_clean_csv(FILE_PATH, CLEANED_FILE_PATH)
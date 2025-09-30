
import pandas as pd
import os
import logging
from tqdm import tqdm

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# --- Caminhos dos Arquivos ---
ORIGINAL_FILE_PATH = r"C:\SCRIPTS\ITS_Article\unified_data_2010_2024.csv"
CONVERTED_FILE_PATH = r"C:\SCRIPTS\ITS_Article\unified_data_UTF8.csv"

def convert_csv_to_utf8_with_progress(original_path, new_path):
    """
    Lê um arquivo CSV em latin-1 e o salva como UTF-8, mostrando uma barra de progresso.
    """
    if not os.path.exists(original_path):
        logging.error(f"ARQUIVO DE ENTRADA NÃO ENCONTRADO: {original_path}")
        return

    logging.info(f"Iniciando a conversão do arquivo: {original_path}")
    
    try:
        # --- ETAPA 1: Preparação ---
        # Detecta o separador
        with open(original_path, 'r', encoding='latin-1') as f:
            first_line = f.readline()
            sep = ';' if first_line.count(';') > first_line.count(',') else ','
        logging.info(f"Separador detectado: '{sep}'")

        # Define o tamanho de cada "pedaço" de leitura
        chunk_size = 100000
        
        # Conta o número total de linhas no arquivo para a barra de progresso
        logging.info("Calculando o total de linhas para a barra de progresso...")
        total_lines = sum(1 for line in open(original_path, 'r', encoding='latin-1'))
        total_chunks = (total_lines // chunk_size) + 1

        # --- ETAPA 2: Conversão em Chunks com Barra de Progresso ---
        logging.info("Iniciando o processo de leitura e gravação...")
        
        # Cria o iterador para ler o arquivo em pedaços
        chunk_iterator = pd.read_csv(original_path, sep=sep, encoding='latin-1', low_memory=False, chunksize=chunk_size)
        
        is_first = True
        # Usa tqdm para criar a barra de progresso
        for chunk in tqdm(chunk_iterator, total=total_chunks, desc="Convertendo para UTF-8"):
            if is_first:
                # Salva o primeiro chunk com cabeçalho
                chunk.to_csv(new_path, mode='w', index=False, encoding='utf-8', sep=',')
                is_first = False
            else:
                # Salva os chunks seguintes sem cabeçalho (append)
                chunk.to_csv(new_path, mode='a', header=False, index=False, encoding='utf-8', sep=',')

        print("\n") # Adiciona uma linha em branco após a barra de progresso
        logging.info("="*50)
        logging.info(f"SUCESSO! O arquivo foi convertido e salvo em:")
        logging.info(new_path)
        logging.info("="*50)

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a conversão: {e}")

if __name__ == "__main__":
    convert_csv_to_utf8_with_progress(ORIGINAL_FILE_PATH, CONVERTED_FILE_PATH)
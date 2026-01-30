# 🦅 Design Soluções | Movimentações Clientes

Uma aplicação web completa de **Supply Chain Analytics** desenvolvida em Python com Streamlit. Esta ferramenta automatiza o processo de ETL (Extração, Transformação e Carga), padronização de dados de estoque, enriquecimento com dimensões e análise visual avançada.

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)

## 🎯 Funcionalidades

A aplicação guia o usuário através de um fluxo de trabalho em 5 etapas ("Wizard Flow"):

1.  **Configuração Inicial:** Upload de amostra para identificar a estrutura do arquivo.
2.  **Mapeamento Inteligente:** Interface visual para mapear colunas de origem (Excel/CSV) para o padrão do sistema (Depósito, SKU, Data, Quantidade, etc.).
3.  **Processamento em Lote (ETL):** * Leitura de múltiplos arquivos massivos.
    * Cruzamento (Join) com tabelas dimensão de **SKU** e **Depósito**.
    * Cálculos estatísticos automáticos (Média, Desvio Padrão, Percentis).
4.  **Dashboard Interativo:**
    * KPIs dinâmicos (Big Numbers).
    * Tabela com suporte a *Drill-down* (clique na linha para filtrar).
    * Gráficos de tendência temporal.
    * **Heatmap "GitHub Style":** Visualização de intensidade de movimentação por semana do ano.
5.  **Exportação:** Download dos dados tratados e analíticos em Excel (.xlsx) ou CSV.

## 🛠️ Tecnologias Utilizadas

* **[Streamlit](https://streamlit.io/):** Framework para interface web interativa.
* **[Polars](https://pola.rs/):** Processamento de dados de alta performance (alternativa rápida ao Pandas).
* **[Pandas](https://pandas.pydata.org/):** Manipulação de datas e compatibilidade legado.
* **[Plotly](https://plotly.com/python/):** Gráficos interativos e responsivos.

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual.

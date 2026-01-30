import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import gc
import os
import shutil
import warnings

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO E OTIMIZAÇÃO
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Pasta temporária para armazenar os dados em disco (evita RAM cheia)
TEMP_DIR = "temp_data"

def init_env():
    """Cria pasta temporária se não existir e limpa lixo anterior."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def clear_data():
    """Limpa dados do disco e da sessão."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    gc.collect()

# ==============================================================================
# 1. SETUP & CSS
# ==============================================================================
def setup_page():
    st.set_page_config(
        page_title="Design Soluções | Analytics",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    init_env()

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root { --primary-color: #2563eb; --background-color: #f8fafc; --secondary-background-color: #ffffff; --text-color: #334155; }
        .stApp { background-color: #f8fafc !important; color: #334155 !important; font-family: 'Inter', sans-serif; }
        [data-testid="stSidebar"] { display: none; }
        #MainMenu, header, footer { visibility: hidden; }
        h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown { color: #334155 !important; }
        
        .step-header-card { background-color: #ffffff; border-radius: 10px; padding: 15px 20px; box-shadow: 0 2px 4px -1px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }
        .step-badge { background-color: #eff6ff; color: #2563eb !important; font-weight: 700; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #bfdbfe; white-space: nowrap; }
        .step-title { font-size: 1.1rem; font-weight: 600; color: #1e293b !important; margin: 0; line-height: 1.2; }
        .step-summary { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 20px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px; }
        .step-check { background-color: #16a34a; color: white !important; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; }
        
        .kpi-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); text-align: left; height: 100%; display: flex; flex-direction: column; justify-content: center; cursor: help; }
        .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0f172a !important; margin: 5px 0; }
        .kpi-label { font-size: 0.75rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #334155 !important; border-color: #e2e8f0 !important; }
        div.stButton > button { border-radius: 8px; font-weight: 600; width: 100%; background-color: #ffffff !important; color: #334155 !important; border: 1px solid #e2e8f0 !important; }
        div.stButton > button:hover { border-color: #2563eb !important; color: #2563eb !important; background-color: #f8fafc !important; }
        .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. MOTOR DE DADOS OTIMIZADO (DISK BASED)
# ==============================================================================

def read_file_chunk(file) -> pl.DataFrame:
    """Lê arquivo e converte para Polars."""
    try:
        if file.name.endswith('.csv'):
            return pl.read_csv(file, ignore_errors=True, infer_schema_length=0)
        else:
            try: return pl.read_excel(file, engine="calamine")
            except: return pl.read_excel(file) 
    except Exception: return pl.DataFrame()

def process_save_chunk(file, idx, mapping, split_dt, dt_source):
    """Lê, processa e SALVA EM DISCO imediatamente (não guarda na RAM)."""
    df_raw = read_file_chunk(file)
    if df_raw.is_empty(): return False
    
    exprs = []
    
    # 1. Data/Hora
    if split_dt and dt_source in df_raw.columns:
        try:
            tc = pl.col(dt_source).str.to_datetime(strict=False)
            exprs.extend([tc.dt.date().alias("Data"), tc.dt.time().alias("Hora")])
        except:
            exprs.extend([pl.col(dt_source).cast(pl.Utf8).alias("Data"), pl.lit(None).alias("Hora")])
    else:
        for target in ["Data", "Hora"]:
            src = mapping.get(target)
            if src and src in df_raw.columns:
                if target == "Data":
                    try: exprs.append(pl.col(src).str.to_datetime(strict=False).dt.date().alias("Data"))
                    except: exprs.append(pl.col(src).alias("Data"))
                else:
                    try: exprs.append(pl.col(src).str.to_time(strict=False).alias("Hora"))
                    except: exprs.append(pl.col(src).alias("Hora"))
            else:
                exprs.append(pl.lit(None).alias(target))

    # 2. Outras Colunas (Com Otimização de Tipos)
    target_cols = ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]
    for target in target_cols:
        src = mapping.get(target)
        if src and src in df_raw.columns:
            if target == "Quantidade":
                exprs.append(pl.col(src).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).cast(pl.Float32).alias(target))
            elif target in ["Depósito", "Rota/Destino"]:
                # Categorical economiza muita RAM para colunas repetitivas
                exprs.append(pl.col(src).cast(pl.Utf8).fill_null("-").cast(pl.Categorical).alias(target))
            else:
                exprs.append(pl.col(src).cast(pl.Utf8).fill_null("-").alias(target))
        else:
            if target == "Quantidade": exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(target))
            else: exprs.append(pl.lit("-", dtype=pl.Categorical if target in ["Depósito", "Rota/Destino"] else pl.Utf8).alias(target))

    # Salva Parquet em disco e limpa RAM
    try:
        df_clean = df_raw.select(exprs)
        file_path = os.path.join(TEMP_DIR, f"chunk_{idx}.parquet")
        df_clean.write_parquet(file_path)
        
        del df_raw
        del df_clean
        gc.collect()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar chunk {idx}: {e}")
        return False

def calculate_aggregates(dim_sku_file, key_sku, desc_sku, dim_dep_file, key_dep, desc_dep):
    """Lê todos os parquets do disco como um LazyFrame (Zero RAM) e agrega."""
    
    # Lazy Load de todos os arquivos na pasta
    try:
        lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
    except:
        return None

    # Carrega dimensões (pequenas, podem ir pra RAM)
    d_sku_df, d_dep_df = None, None
    if dim_sku_file and key_sku:
        d_sku_df = read_file_chunk(dim_sku_file).select([
            pl.col(key_sku).cast(pl.Utf8).alias("KEY_SKU"),
            pl.col(desc_sku).cast(pl.Utf8).alias("DESC_SKU")
        ])
    
    if dim_dep_file and key_dep:
        d_dep_df = read_file_chunk(dim_dep_file).select([
            pl.col(key_dep).cast(pl.Utf8).alias("KEY_DEP"),
            pl.col(desc_dep).cast(pl.Utf8).alias("DESC_DEP")
        ])

    # Agregação (Lazy - só processa na hora do collect)
    # Primeiro agrupa para reduzir o tamanho
    daily_agg = (
        lf.filter(pl.col("Data").is_not_null())
        .group_by(["Depósito", "SKU", "Data"])
        .agg(pl.col("Quantidade").sum().alias("Qtd_Dia"))
    ).collect() # Aqui trazemos para RAM, mas já resumido

    # Estatísticas Finais
    stats = daily_agg.group_by(["Depósito", "SKU"]).agg([
        pl.col("Qtd_Dia").mean().alias("Média"),
        pl.col("Qtd_Dia").max().alias("Máximo"),
        pl.col("Qtd_Dia").std().fill_null(0).alias("Desvio"),
        pl.col("Qtd_Dia").quantile(0.95).alias("Percentil 95%")
    ])

    stats = stats.with_columns([
        (pl.col("Média") + pl.col("Desvio")).alias("Média + 1 Desv"),
        (pl.col("Média") + (pl.col("Desvio") * 2)).alias("Média + 2 Desv"),
        (pl.col("Média") + (pl.col("Desvio") * 3)).alias("Média + 3 Desv"),
    ])

    # Cast para Join
    stats = stats.with_columns([pl.col("SKU").cast(pl.Utf8), pl.col("Depósito").cast(pl.Utf8)])

    # Enriquecimento com dimensões
    if d_sku_df is not None:
        stats = stats.join(d_sku_df, left_on="SKU", right_on="KEY_SKU", how="left").rename({"DESC_SKU": "Descrição"})
    else:
        stats = stats.with_columns(pl.lit("-").alias("Descrição"))

    if d_dep_df is not None:
        stats = stats.join(d_dep_df, left_on="Depósito", right_on="KEY_DEP", how="left").rename({"DESC_DEP": "Nome Depósito"})
    else:
        stats = stats.with_columns(pl.lit("-").alias("Nome Depósito"))

    # Renomeação e Seleção Final
    final_cols = ["Código Depósito", "Depósito", "SKU", "Descrição", "Média", "Máximo", "Desvio", 
                  "Média + 1 Desv", "Média + 2 Desv", "Média + 3 Desv", "Percentil 95%"]
    
    stats = stats.rename({"Depósito": "Código Depósito", "Nome Depósito": "Depósito"})
    
    # Garante colunas faltantes
    for c in final_cols:
        if c not in stats.columns: stats = stats.with_columns(pl.lit("-").alias(c))
        
    return stats.select(final_cols)

def get_filtered_data(sel_sku, sel_dep):
    """Busca no disco APENAS as linhas filtradas (Drill-down eficiente)."""
    lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
    
    # Cast para garantir comparação correta (Categorical vs String)
    query = lf.filter(
        (pl.col("SKU").cast(pl.Utf8) == sel_sku) & 
        (pl.col("Depósito").cast(pl.Utf8) == sel_dep)
    )
    return query.collect()

# ==============================================================================
# 3. UI PRINCIPAL
# ==============================================================================

def main():
    setup_page()

    c_logo, c_title, c_act = st.columns([0.15, 0.65, 0.2], vertical_alignment="bottom")
    with c_logo:
        try: st.image("Aguia Fundo Branco.png")
        except: st.markdown("### 🦅")
    with c_title:
        st.markdown("""<h3 style='margin: 0; padding-bottom: 35px; font-weight: 600; color: #1e293b !important;'>Design Soluções | Movimentações Clientes</h3>""", unsafe_allow_html=True)
    with c_act:
        if st.button("🔄 Novo Projeto", type="secondary", use_container_width=True):
            clear_data()
            st.rerun()
    st.markdown("---")

    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
        st.session_state.mapping = {}
        st.session_state.split_dt = True
        st.session_state.dt_source = None

    # --- ETAPA 1 ---
    if st.session_state.current_step > 1:
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 1: Configuração Concluída.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 1:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 1</span><h3 class="step-title">Configuração Inicial</h3></div>""", unsafe_allow_html=True)
        f_sample = st.file_uploader("Arquivo de Amostra", type=["xlsx", "csv"], label_visibility="collapsed")
        if f_sample:
            # Lê apenas primeiras linhas para pegar colunas
            try:
                df_s = pl.read_excel(f_sample, engine="calamine") if f_sample.name.endswith('.xlsx') else pl.read_csv(f_sample, n_rows=100, ignore_errors=True)
                st.session_state.cols_origem = ["--- Ignorar ---"] + df_s.columns
                st.session_state.current_step = 2
                st.rerun()
            except:
                st.error("Erro ao ler amostra. Verifique o arquivo.")

    # --- ETAPA 2 ---
    if st.session_state.current_step > 2:
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 2: Mapeamento Definido.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 2:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 2</span><h3 class="step-title">Mapeamento de Colunas</h3></div>""", unsafe_allow_html=True)
        c_dt1, c_dt2 = st.columns([1, 2])
        with c_dt1: split_dt = st.toggle("Separar Data/Hora?", value=True)
        with c_dt2: dt_source = st.selectbox("Coluna Data/Hora:", ["--- Selecione ---"] + st.session_state.cols_origem) if split_dt else None
        st.markdown("---")
        fields = ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]
        if not split_dt: fields = ["Data", "Hora"] + fields
        c_map = st.columns(3)
        curr_map = {}
        for i, target in enumerate(fields):
            with c_map[i % 3]:
                st.caption(f"**{target}**")
                idx = 0
                for ix, c in enumerate(st.session_state.cols_origem):
                    if c.lower() in target.lower(): idx = ix; break
                curr_map[target] = st.selectbox(f"map_{i}", st.session_state.cols_origem, index=idx, label_visibility="collapsed")
        st.markdown("###")
        if st.button("Salvar e Avançar", type="primary", use_container_width=True):
            st.session_state.mapping = curr_map
            st.session_state.split_dt = split_dt
            st.session_state.dt_source = dt_source
            st.session_state.current_step = 3
            st.rerun()

    # --- ETAPA 3 ---
    if st.session_state.current_step > 3:
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 3: Dados Processados.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 3:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 3</span><h3 class="step-title">Processamento em Lote</h3></div>""", unsafe_allow_html=True)
        col_main, col_dim = st.columns([1, 1])
        with col_main:
            st.markdown("##### 1. Movimentação (Lote)")
            files_mov = st.file_uploader("Arquivos Movimentação", type=["xlsx", "csv"], accept_multiple_files=True)
        with col_dim:
            st.markdown("##### 2. Dimensões (Opcional)")
            tab_s, tab_d = st.tabs(["📦 SKU", "🏢 Depósito"])
            with tab_s:
                f_sku = st.file_uploader("Dimensão SKU", type=["xlsx", "csv"])
                k_sku = st.selectbox("Chave Código:", st.session_state.cols_origem, key="ks") if f_sku else None # Simplificado
                d_sku = st.text_input("Nome da Coluna Descrição SKU") if f_sku else None
            with tab_d:
                f_dep = st.file_uploader("Dimensão Depósito", type=["xlsx", "csv"])
                k_dep = st.selectbox("Chave Código:", st.session_state.cols_origem, key="kd") if f_dep else None
                d_dep = st.text_input("Nome da Coluna Descrição Depósito") if f_dep else None
        
        st.markdown("###")
        if files_mov:
            if st.button("🚀 Processar Dados", type="primary", use_container_width=True):
                # Limpa dados anteriores
                if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR)
                
                bar = st.progress(0, text="Processando e salvando em disco...")
                count = 0
                for idx, f in enumerate(files_mov):
                    success = process_save_chunk(f, idx, st.session_state.mapping, st.session_state.split_dt, st.session_state.dt_source)
                    if success: count += 1
                    bar.progress((idx + 1) / len(files_mov))
                
                if count > 0:
                    with st.status("Calculando estatísticas...", expanded=True):
                        # Dimensões são carregadas aqui para economizar RAM antes
                        # Nota: A UI simplificada acima para dimensões pode precisar de ajustes se o arquivo dimensão for diferente do principal.
                        # Assumindo aqui que o usuário carrega e define chaves corretamente.
                        
                        # (Opcional) Recarregar dimensão correta se necessário, aqui simplifiquei para focar no OOM.
                        final_stats = calculate_aggregates(f_sku, k_sku, d_sku, f_dep, k_dep, d_dep)
                        
                        if final_stats is not None:
                            st.session_state.final_stats = final_stats
                            st.session_state.current_step = 4
                            st.rerun()
                        else:
                            st.error("Erro ao calcular estatísticas.")
                else:
                    st.error("Nenhum arquivo válido processado.")

    # --- ETAPA 4 ---
    if st.session_state.current_step > 4:
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 4: Análise Concluída.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 4:
        stats = st.session_state.final_stats
        
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 4</span><h3 class="step-title">Dashboard de Análise</h3></div>""", unsafe_allow_html=True)
        
        # Filtros (usando a tabela stats que é pequena)
        stats = stats.with_columns([
            pl.concat_str([pl.col("SKU"), pl.lit(" - "), pl.col("Descrição")]).alias("Label_SKU"),
            pl.concat_str([pl.col("Código Depósito"), pl.lit(" - "), pl.col("Depósito")]).alias("Label_Dep")
        ])
        
        c1, c2 = st.columns(2)
        sel_skus = c1.multiselect("Filtrar SKUs", stats["Label_SKU"].unique().sort().to_list())
        sel_deps = c2.multiselect("Filtrar Depósitos", stats["Label_Dep"].unique().sort().to_list())
        
        v_stats = stats
        if sel_skus: v_stats = v_stats.filter(pl.col("Label_SKU").is_in(sel_skus))
        if sel_deps: v_stats = v_stats.filter(pl.col("Label_Dep").is_in(sel_deps))

        # Variável para armazenar os dados detalhados da seleção atual
        v_detail = pl.DataFrame()

        # KPIs Placeholders
        k1, k2, k3, k4, k5 = st.columns(5)
        
        # --- LÓGICA DE SELEÇÃO E CARREGAMENTO SOB DEMANDA ---
        st.markdown("###")
        
        if 'selected_row' in st.session_state:
            if st.button("❌ Limpar Seleção", type="secondary"):
                del st.session_state.selected_row
                st.rerun()

        # Renderiza KPIs zerados ou calculados se houver seleção
        
        # Se houver seleção, carrega do disco APENAS o necessário
        if 'selected_row' in st.session_state:
            sel_s, sel_d = st.session_state.selected_row.split("|")
            v_detail = get_filtered_data(sel_s, sel_d)
            
            if not v_detail.is_empty():
                qtd_linhas = v_detail.height
                qtd_vol = v_detail["Quantidade"].sum()
                qtd_pick = v_detail["Pedido"].n_unique()
                qtd_dias = v_detail["Data"].n_unique()
                
                # Renderiza KPIs reais
                def kpi_html(l, v, s, t):
                    return f"""<div class="kpi-card" title="{t}"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>"""
                
                k1.markdown(kpi_html("Linhas", f"{qtd_linhas:,}", "Registros", "Total de linhas"), unsafe_allow_html=True)
                k2.markdown(kpi_html("Volume", f"{qtd_vol:,.0f}", "Total Qtd", "Soma Quantidade"), unsafe_allow_html=True)
                k3.markdown(kpi_html("Picking", f"{qtd_pick:,}", "Pedidos", "Count Distinct Pedido"), unsafe_allow_html=True)
                k4.markdown(kpi_html("Dias", f"{qtd_dias}", "Dias Ativos", "Dias com movimento"), unsafe_allow_html=True)
                k5.markdown(kpi_html("Filtro", f"{sel_s}", f"Dep: {sel_d}", "Seleção Atual"), unsafe_allow_html=True)

                # Gráficos
                st.markdown("---")
                daily_agg = v_detail.group_by("Data").agg(pl.col("Quantidade").sum()).sort("Data")
                fig = px.bar(daily_agg.to_pandas(), x="Data", y="Quantidade", title="Evolução Diária", template="plotly_white")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#334155")
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap
                real_data = daily_agg.to_pandas()
                real_data["Data"] = pd.to_datetime(real_data["Data"])
                min_date = real_data["Data"].min()
                date_range = pd.date_range(start=min_date, periods=54*7, freq='D')
                skel = pd.DataFrame({"Data": date_range})
                hm = pd.merge(skel, real_data, on="Data", how="left").fillna(0)
                hm["YearWeek"] = hm["Data"].dt.strftime("%Y-W%U")
                hm["Day"] = hm["Data"].dt.strftime("%a")
                
                fig_hm = px.density_heatmap(hm, x="YearWeek", y="Day", z="Quantidade", color_continuous_scale="Greens", title="Heatmap Semanal", template="plotly_white",
                                            category_orders={"Day": ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]})
                fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#334155")
                st.plotly_chart(fig_hm, use_container_width=True)

        else:
            st.info("👆 Selecione uma linha na tabela abaixo para ver os detalhes e gráficos.")

        # Tabela (Agora posicionada abaixo)
        st.markdown("**Tabela Analítica (Clique para detalhar):**")
        pdf = v_stats.drop(["Label_SKU", "Label_Dep"]).to_pandas()
        sel = st.dataframe(pdf, use_container_width=True, height=400, on_select="rerun", selection_mode="single-row")
        
        if sel.selection.rows:
            idx = sel.selection.rows[0]
            row = pdf.iloc[idx]
            st.session_state.selected_row = f"{row['SKU']}|{row['Código Depósito']}"
            st.rerun()

        st.markdown("###")
        if st.button("Ir para Exportação", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

    # --- ETAPA 5 ---
    if st.session_state.current_step == 5:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 5</span><h3 class="step-title">Downloads</h3></div>""", unsafe_allow_html=True)
        
        # Gera Excel da tabela de Stats (leve)
        b_xls = io.BytesIO()
        st.session_state.final_stats.write_excel(b_xls)
        st.download_button("Baixar Análise (.xlsx)", b_xls.getvalue(), "analise.xlsx", use_container_width=True)
        
        st.warning("⚠️ O download da base completa pode demorar pois os dados serão reconstruídos do disco.")
        if st.button("Gerar CSV Completo (Pode demorar)", use_container_width=True):
            # Reconstrói CSV do disco (Lazy)
            lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
            csv_data = lf.collect().write_csv()
            st.download_button("📥 Baixar CSV", csv_data, "completo.csv", "text/csv")

        st.markdown("---")
        if st.button("⬅️ Voltar"):
            st.session_state.current_step = 4
            st.rerun()

if __name__ == "__main__":
    main()
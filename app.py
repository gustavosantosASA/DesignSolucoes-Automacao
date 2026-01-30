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

TEMP_DIR = "temp_data"

def init_env():
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

def clear_data():
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    for key in list(st.session_state.keys()): del st.session_state[key]
    gc.collect()

# ==============================================================================
# 1. SETUP & CSS
# ==============================================================================
def setup_page():
    st.set_page_config(page_title="Design Soluções | Analytics", page_icon="🦅", layout="wide", initial_sidebar_state="collapsed")
    init_env()
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root { --primary-color: #2563eb; --background-color: #f8fafc; --secondary-background-color: #ffffff; --text-color: #334155; }
        .stApp { background-color: #f8fafc !important; color: #334155 !important; font-family: 'Inter', sans-serif; }
        [data-testid="stSidebar"] { display: none; }
        #MainMenu, header, footer { visibility: hidden; }
        h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown { color: #334155 !important; }
        .stMarkdown h3 { color: #1e293b !important; }
        .step-header-card { background-color: #ffffff; border-radius: 10px; padding: 15px 20px; box-shadow: 0 2px 4px -1px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }
        .step-badge { background-color: #eff6ff; color: #2563eb !important; font-weight: 700; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #bfdbfe; white-space: nowrap; }
        .step-title { font-size: 1.1rem; font-weight: 600; color: #1e293b !important; margin: 0; line-height: 1.2; }
        .step-summary { background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 20px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px; }
        .step-check { background-color: #16a34a; color: white !important; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; }
        .kpi-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); text-align: left; height: 100%; display: flex; flex-direction: column; justify-content: center; cursor: help; }
        .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0f172a !important; margin: 5px 0; }
        .kpi-label { font-size: 0.75rem; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-sub { font-size: 0.7rem; color: #94a3b8 !important; margin-top: 2px; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #334155 !important; border-color: #e2e8f0 !important; }
        ul[data-baseweb="menu"] { background-color: #ffffff !important; }
        [data-testid="stFileUploadDropzone"] { background-color: #ffffff !important; border-color: #e2e8f0 !important; }
        [data-testid="stDataFrame"] { background-color: #ffffff !important; }
        div.stButton > button { border-radius: 8px; font-weight: 600; width: 100%; background-color: #ffffff !important; color: #334155 !important; border: 1px solid #e2e8f0 !important; }
        div.stButton > button:hover { border-color: #2563eb !important; color: #2563eb !important; background-color: #f8fafc !important; }
        .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. MOTOR DE DADOS (DISK BASED)
# ==============================================================================

def read_file_chunk(file) -> pl.DataFrame:
    if hasattr(file, 'seek'): file.seek(0)
    try:
        if file.name.endswith('.csv'): return pl.read_csv(file, ignore_errors=True, infer_schema_length=0)
        else:
            try: return pl.read_excel(file, engine="calamine")
            except: 
                file.seek(0)
                return pl.from_pandas(pd.read_excel(file))
    except: return pl.DataFrame()

def load_sample_optimized(file) -> pl.DataFrame:
    if hasattr(file, 'seek'): file.seek(0)
    try:
        if file.name.endswith('.csv'): return pl.read_csv(file, n_rows=100, ignore_errors=True)
        else: 
            try: return pl.read_excel(file, engine="calamine")
            except: return pl.read_excel(file) 
    except: return pl.DataFrame()

def process_save_chunk(file, idx, mapping, split_dt, dt_source):
    df_raw = read_file_chunk(file)
    if df_raw.is_empty(): return False
    
    exprs = []
    # Data/Hora
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

    # Colunas Gerais
    target_cols = ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]
    for target in target_cols:
        src = mapping.get(target)
        if src and src in df_raw.columns:
            if target == "Quantidade":
                exprs.append(pl.col(src).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).cast(pl.Float32).alias(target))
            else:
                exprs.append(pl.col(src).cast(pl.Utf8, strict=False).fill_null("").alias(target))
        else:
            if target == "Quantidade": exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(target))
            else: exprs.append(pl.lit("", dtype=pl.Utf8).alias(target))

    try:
        df_clean = df_raw.select(exprs)
        file_path = os.path.join(TEMP_DIR, f"chunk_{idx}.parquet")
        df_clean.write_parquet(file_path)
        del df_raw, df_clean
        gc.collect()
        return True
    except: return False

def calculate_stats_table(dim_sku_file, key_sku, desc_sku, dim_dep_file, key_dep, desc_dep):
    try: lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
    except: return None

    # Agregação para Tabela de Stats
    daily_agg = lf.filter(pl.col("Data").is_not_null()).group_by(["Depósito", "SKU", "Data"]).agg(pl.col("Quantidade").sum().alias("Qtd_Dia")).collect()
    
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

    # Join Dimensões
    if dim_sku_file:
        d_sku = read_file_chunk(dim_sku_file).select([pl.col(key_sku).cast(pl.Utf8).alias("K"), pl.col(desc_sku).cast(pl.Utf8).alias("D")])
        stats = stats.join(d_sku, left_on="SKU", right_on="K", how="left").rename({"D": "Descrição"})
    else: stats = stats.with_columns(pl.lit("-").alias("Descrição"))

    if dim_dep_file:
        d_dep = read_file_chunk(dim_dep_file).select([pl.col(key_dep).cast(pl.Utf8).alias("K"), pl.col(desc_dep).cast(pl.Utf8).alias("D")])
        stats = stats.join(d_dep, left_on="Depósito", right_on="K", how="left").rename({"D": "Nome Depósito"})
    else: stats = stats.with_columns(pl.lit("-").alias("Nome Depósito"))

    stats = stats.rename({"Depósito": "Código Depósito", "Nome Depósito": "Depósito", "SKU": "SKU", "Descrição": "Descrição"})
    
    cols = ["Código Depósito", "Depósito", "SKU", "Descrição", "Média", "Máximo", "Desvio", "Média + 1 Desv", "Média + 2 Desv", "Média + 3 Desv", "Percentil 95%"]
    for c in cols: 
        if c not in stats.columns: stats = stats.with_columns(pl.lit("-").alias(c))
    
    return stats.select(cols)

def get_dashboard_metrics(sel_skus, sel_deps, drill_sku=None, drill_dep=None):
    """
    Calcula métricas e dados para gráficos DIRETAMENTE DO DISCO.
    Não carrega linhas individuais, apenas agregados = Baixa Memória.
    """
    lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
    
    # Aplica filtros globais (Multiselects)
    if sel_skus: lf = lf.filter(pl.col("SKU").cast(pl.Utf8).is_in(sel_skus))
    if sel_deps: lf = lf.filter(pl.col("Depósito").cast(pl.Utf8).is_in(sel_deps))
    
    # Aplica Drill Down (Se houver clique na tabela)
    if drill_sku and drill_dep:
        lf = lf.filter((pl.col("SKU") == drill_sku) & (pl.col("Depósito") == drill_dep))

    # 1. KPIs Escalares (Otimizado)
    # count() é rápido. sum() é rápido. n_unique() pode ser pesado, usamos approx se necessário, mas aqui vamos tentar exato.
    kpis = lf.select([
        pl.len().alias("lines"),
        pl.col("Quantidade").sum().alias("vol"),
        pl.col("Pedido").n_unique().alias("pick"),
        pl.col("SKU").n_unique().alias("skus"),
        pl.col("Depósito").n_unique().alias("deps"),
        pl.col("Data").n_unique().alias("days")
    ]).collect().row(0) # Retorna tupla (lines, vol, pick, skus, deps, days)

    # 2. Dados Diários para Gráficos (Agrupado = Pequeno na RAM)
    daily_agg = (
        lf.filter(pl.col("Data").is_not_null())
        .group_by("Data")
        .agg(pl.col("Quantidade").sum())
        .sort("Data")
    ).collect()

    return kpis, daily_agg

# ==============================================================================
# 3. UI PRINCIPAL
# ==============================================================================
def main():
    setup_page()

    c_logo, c_title, c_act = st.columns([0.15, 0.65, 0.2], vertical_alignment="bottom")
    with c_logo:
        try: st.image("Aguia Fundo Branco.png")
        except: st.markdown("### 🦅")
    with c_title: st.markdown("""<h3 style='margin: 0; padding-bottom: 35px; font-weight: 600; color: #1e293b !important;'>Design Soluções | Movimentações Clientes</h3>""", unsafe_allow_html=True)
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

    # ETAPA 1
    if st.session_state.current_step > 1: st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 1: Configuração Concluída.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 1:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 1</span><h3 class="step-title">Configuração Inicial</h3></div>""", unsafe_allow_html=True)
        f = st.file_uploader("Arquivo de Amostra", type=["xlsx", "csv"], label_visibility="collapsed")
        if f:
            df_s = load_sample_optimized(f)
            st.session_state.cols_origem = ["--- Ignorar ---"] + df_s.columns
            st.session_state.current_step = 2
            st.rerun()

    # ETAPA 2
    if st.session_state.current_step > 2: st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 2: Mapeamento Definido.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 2:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 2</span><h3 class="step-title">Mapeamento de Colunas</h3></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1: split_dt = st.toggle("Separar Data/Hora?", value=True)
        with c2: dt_source = st.selectbox("Coluna Data/Hora:", ["--- Selecione ---"] + st.session_state.cols_origem) if split_dt else None
        st.markdown("---")
        fields = ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]
        if not split_dt: fields = ["Data", "Hora"] + fields
        c_map = st.columns(3)
        curr_map = {}
        for i, target in enumerate(fields):
            with c_map[i % 3]:
                idx = 0
                for ix, c in enumerate(st.session_state.cols_origem):
                    if c.lower() in target.lower(): idx = ix; break
                curr_map[target] = st.selectbox(target, st.session_state.cols_origem, index=idx)
        st.markdown("###")
        if st.button("Salvar e Avançar", type="primary", use_container_width=True):
            st.session_state.mapping = curr_map
            st.session_state.split_dt = split_dt
            st.session_state.dt_source = dt_source
            st.session_state.current_step = 3
            st.rerun()

    # ETAPA 3
    if st.session_state.current_step > 3: st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 3: Dados Processados.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 3:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 3</span><h3 class="step-title">Processamento em Lote</h3></div>""", unsafe_allow_html=True)
        cm, cd = st.columns([1, 1])
        with cm:
            st.markdown("##### 1. Movimentação")
            files_mov = st.file_uploader("Arquivos", type=["xlsx", "csv"], accept_multiple_files=True)
        with cd:
            st.markdown("##### 2. Dimensões")
            ts, td = st.tabs(["📦 SKU", "🏢 Depósito"])
            with ts:
                f_sku = st.file_uploader("Dimensão SKU", type=["xlsx", "csv"])
                k_sku, d_sku = None, None
                if f_sku:
                    cols = load_sample_optimized(f_sku).columns
                    k_sku = st.selectbox("Chave Código:", cols, key="ks")
                    d_sku = st.selectbox("Col. Descrição:", cols, key="ds")
            with td:
                f_dep = st.file_uploader("Dimensão Depósito", type=["xlsx", "csv"])
                k_dep, d_dep = None, None
                if f_dep:
                    cols = load_sample_optimized(f_dep).columns
                    k_dep = st.selectbox("Chave Código:", cols, key="kd")
                    d_dep = st.selectbox("Col. Descrição:", cols, key="dd")
        st.markdown("###")
        if files_mov and st.button("🚀 Processar", type="primary", use_container_width=True):
            if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            bar = st.progress(0, "Processando...")
            for i, f in enumerate(files_mov):
                process_save_chunk(f, i, st.session_state.mapping, st.session_state.split_dt, st.session_state.dt_source)
                bar.progress((i+1)/len(files_mov))
            
            with st.status("Calculando estatísticas...", expanded=True):
                stats = calculate_stats_table(f_sku, k_sku, d_sku, f_dep, k_dep, d_dep)
                if stats is not None:
                    st.session_state.final_stats = stats
                    st.session_state.current_step = 4
                    st.rerun()
                else: st.error("Erro no cálculo.")

    # ETAPA 4 (DASHBOARD INTELIGENTE)
    if st.session_state.current_step == 4:
        stats = st.session_state.final_stats
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 4</span><h3 class="step-title">Dashboard de Análise</h3></div>""", unsafe_allow_html=True)
        
        # Filtros
        stats = stats.with_columns([
            pl.concat_str([pl.col("SKU"), pl.lit(" - "), pl.col("Descrição")]).alias("Label_SKU"),
            pl.concat_str([pl.col("Código Depósito"), pl.lit(" - "), pl.col("Depósito")]).alias("Label_Dep")
        ])
        c1, c2 = st.columns(2)
        sel_skus = c1.multiselect("Filtrar SKUs", stats["Label_SKU"].unique().sort().to_list())
        sel_deps = c2.multiselect("Filtrar Depósitos", stats["Label_Dep"].unique().sort().to_list())
        
        # Filtra tabela
        v_stats = stats
        if sel_skus: v_stats = v_stats.filter(pl.col("Label_SKU").is_in(sel_skus))
        if sel_deps: v_stats = v_stats.filter(pl.col("Label_Dep").is_in(sel_deps))

        # Determina contexto (Global ou Drill Down)
        drill_sku, drill_dep = None, None
        
        if 'selected_row' in st.session_state:
            if st.button("❌ Limpar Seleção (Voltar ao Global)", type="secondary"):
                del st.session_state.selected_row
                st.rerun()
            drill_sku, drill_dep = st.session_state.selected_row.split("|")

        # Filtros códigos puros para passar ao motor
        filter_sku_codes = [s.split(" - ")[0] for s in sel_skus] if sel_skus else None
        filter_dep_codes = [d.split(" - ")[0] for d in sel_deps] if sel_deps else None

        # --- BUSCA DADOS AGREGADOS DO DISCO (Zero RAM Risk) ---
        kpi_vals, daily_agg = get_dashboard_metrics(filter_sku_codes, filter_dep_codes, drill_sku, drill_dep)
        
        # KPIs
        lines, vol, picks, skus, deps, days = kpi_vals
        avg_day = vol / days if days > 0 else 0
        max_day = daily_agg["Quantidade"].max() if not daily_agg.is_empty() else 0

        def kpi_html(l, v, s, t): return f"""<div class="kpi-card" title="{t}"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>"""
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_html("Linhas", f"{lines:,}".replace(",", "."), "Total", "Registros filtrados"), unsafe_allow_html=True)
        k2.markdown(kpi_html("Volume", f"{vol:,.0f}".replace(",", "."), "Unidades", "Soma Quantidade"), unsafe_allow_html=True)
        k3.markdown(kpi_html("Picking", f"{picks:,}".replace(",", "."), "Pedidos", "Pedidos Únicos"), unsafe_allow_html=True)
        k4.markdown(kpi_html("SKUs", f"{skus:,}", "Produtos", "SKUs Distintos"), unsafe_allow_html=True)
        k5.markdown(kpi_html("Dias", f"{days}", "Ativos", "Dias com movimento"), unsafe_allow_html=True)
        
        st.markdown("###")
        kt1, kt2, kt3 = st.columns(3)
        kt1.markdown(kpi_html("Depósitos", f"{deps}", "Locais", "Depósitos Distintos"), unsafe_allow_html=True)
        kt2.markdown(kpi_html("Média Diária", f"{avg_day:,.0f}".replace(",", "."), "Unid/Dia", "Volume / Dias"), unsafe_allow_html=True)
        kt3.markdown(kpi_html("Pico Máximo", f"{max_day:,.0f}".replace(",", "."), "Recorde", "Maior dia"), unsafe_allow_html=True)

        if not daily_agg.is_empty():
            st.markdown("---")
            pdf = daily_agg.to_pandas()
            # Bar Chart
            fig = px.bar(pdf, x="Data", y="Quantidade", title="Evolução Temporal", template="plotly_white")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#334155")
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            pdf["Data"] = pd.to_datetime(pdf["Data"])
            min_d = pdf["Data"].min()
            dates = pd.date_range(start=min_d, periods=54*7, freq='D')
            skel = pd.DataFrame({"Data": dates})
            hm = pd.merge(skel, pdf, on="Data", how="left").fillna(0)
            hm["W"] = hm["Data"].dt.strftime("%Y-W%U")
            hm["D"] = hm["Data"].dt.strftime("%a")
            
            fig_hm = px.density_heatmap(hm, x="W", y="D", z="Quantidade", color_continuous_scale="Greens", title="Heatmap Sazonalidade", template="plotly_white", category_orders={"D": ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]})
            fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#334155")
            st.plotly_chart(fig_hm, use_container_width=True)

        # TABELA
        st.markdown("###")
        st.markdown("**Tabela Analítica (Clique para detalhar):**")
        pdf_tbl = v_stats.drop(["Label_SKU", "Label_Dep"]).to_pandas()
        sel = st.dataframe(pdf_tbl, use_container_width=True, height=400, on_select="rerun", selection_mode="single-row", column_config={"Média": st.column_config.NumberColumn(format="%.2f")})
        
        if sel.selection.rows:
            row = pdf_tbl.iloc[sel.selection.rows[0]]
            st.session_state.selected_row = f"{row['SKU']}|{row['Código Depósito']}"
            st.rerun()

        if st.button("Ir para Exportação", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

    # ETAPA 5
    if st.session_state.current_step == 5:
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 5</span><h3 class="step-title">Downloads</h3></div>""", unsafe_allow_html=True)
        b = io.BytesIO()
        st.session_state.final_stats.write_excel(b)
        st.download_button("Baixar Análise (.xlsx)", b.getvalue(), "analise.xlsx", use_container_width=True)
        if st.button("Gerar CSV Completo", use_container_width=True):
            lf = pl.scan_parquet(f"{TEMP_DIR}/*.parquet")
            st.download_button("📥 Baixar CSV", lf.collect().write_csv(), "completo.csv", "text/csv")
        st.markdown("---")
        if st.button("⬅️ Voltar"):
            st.session_state.current_step = 4
            st.rerun()

if __name__ == "__main__":
    main()
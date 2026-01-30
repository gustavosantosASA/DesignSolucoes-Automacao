import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import time

# ==============================================================================
# 1. SETUP & CSS
# ==============================================================================

def setup_page():
    st.set_page_config(
        page_title="Design Soluções | Movimentações Clientes",
        page_icon="🦅",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; color: #334155; }
        [data-testid="stSidebar"] { display: none; }
        #MainMenu, header, footer { visibility: hidden; }
        
        /* --- ESTILO DO CABEÇALHO DA ETAPA (CARD) --- */
        .step-header-card {
            background-color: white; 
            border-radius: 10px; 
            padding: 15px 20px; 
            box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.05); 
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
            display: flex; 
            align-items: center; 
            gap: 12px;
        }
        
        .step-badge {
            background-color: #eff6ff; color: #2563eb; font-weight: 700;
            padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid #bfdbfe;
            white-space: nowrap;
        }
        
        .step-title { 
            font-size: 1.1rem; font-weight: 600; color: #1e293b; margin: 0; 
            line-height: 1.2;
        }
        
        /* --- RESUMO DE ETAPA CONCLUÍDA (VERDE) --- */
        .step-summary {
            background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
            padding: 12px 20px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px;
        }
        .step-check {
            background-color: #16a34a; color: white; width: 24px; height: 24px;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 12px;
        }
        .step-text { color: #166534; font-weight: 600; font-size: 0.95rem; margin: 0; }
        
        /* --- KPI CARDS --- */
        .kpi-card {
            background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); text-align: left;
            height: 100%; display: flex; flex-direction: column; justify-content: center;
            transition: all 0.2s ease;
        }
        .kpi-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
        .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 5px 0; }
        .kpi-label { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-sub { font-size: 0.7rem; color: #94a3b8; margin-top: 2px; }

        /* UI ELEMENTS */
        div[data-baseweb="select"] > div { border-radius: 8px; }
        div.stButton > button { border-radius: 8px; font-weight: 600; width: 100%; }
        
        /* Remove padding extra do topo */
        .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. MOTOR DE DADOS
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_sample(file) -> pl.DataFrame:
    try:
        if file.name.endswith('.csv'): return pl.read_csv(file, n_rows=100, ignore_errors=True, try_parse_dates=True)
        else: return pl.read_excel(file) 
    except: return pl.DataFrame()

def load_full_safe(file) -> pl.DataFrame:
    try:
        if file.name.endswith('.csv'):
            return pl.read_csv(file, ignore_errors=True, try_parse_dates=True, infer_schema_length=0)
        else:
            try: return pl.read_excel(file)
            except: 
                file.seek(0)
                return pl.from_pandas(pd.read_excel(file))
    except: return pl.DataFrame()

def load_dim_full(file) -> pl.DataFrame:
    try:
        df = load_full_safe(file)
        if not df.is_empty():
            return df.select([pl.col(c).cast(pl.Utf8) for c in df.columns])
        return df
    except: return pl.DataFrame()

def process_etl_batch(files, mapping, split_dt, dt_source):
    dfs = []
    required_cols = ["Depósito", "SKU", "Pedido", "Caixa", "Data", "Hora", "Quantidade", "Rota/Destino"]
    
    prog_bar = st.progress(0, text="Lendo arquivos...")
    total = len(files)

    for i, f in enumerate(files):
        prog_bar.progress((i+1)/total, text=f"Processando: {f.name}")
        df_raw = load_full_safe(f)
        if df_raw.is_empty(): continue
        
        try:
            exprs = []
            if split_dt and dt_source in df_raw.columns:
                try:
                    tc = pl.col(dt_source).str.to_datetime(strict=False)
                    exprs.extend([tc.dt.date().alias("Data"), tc.dt.time().alias("Hora")])
                except:
                    exprs.extend([pl.col(dt_source).alias("Data"), pl.lit(None).alias("Hora")])
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

            for target in ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]:
                src = mapping.get(target)
                if src and src in df_raw.columns:
                    if target == "Quantidade":
                        exprs.append(pl.col(src).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).alias(target))
                    else:
                        exprs.append(pl.col(src).cast(pl.Utf8, strict=False).alias(target))
                else:
                    exprs.append(pl.lit(None).alias(target))

            df_proc = df_raw.select(exprs)
            missing = [c for c in required_cols if c not in df_proc.columns]
            if missing: df_proc = df_proc.with_columns([pl.lit(None).alias(c) for c in missing])
            
            dfs.append(df_proc.select(required_cols))
        except: continue
        
    prog_bar.empty()
    if not dfs: return pl.DataFrame()
    return pl.concat(dfs, how="vertical")

def enrich_and_calculate_stats(main_df, dim_sku_file, key_sku, desc_sku, dim_dep_file, key_dep, desc_dep):
    res = main_df
    
    # 1. Enriquecimento
    if dim_sku_file and key_sku:
        d_sku = load_dim_full(dim_sku_file)
        if not d_sku.is_empty() and key_sku in d_sku.columns:
            res = res.with_columns(pl.col("SKU").cast(pl.Utf8))
            d_sku = d_sku.with_columns(pl.col(key_sku).cast(pl.Utf8))
            if desc_sku and desc_sku in d_sku.columns: d_sku = d_sku.rename({desc_sku: "SKU_DESC"})
            res = res.join(d_sku, left_on="SKU", right_on=key_sku, how="left", suffix="_sku_dim")

    if dim_dep_file and key_dep:
        d_dep = load_dim_full(dim_dep_file)
        if not d_dep.is_empty() and key_dep in d_dep.columns:
            res = res.with_columns(pl.col("Depósito").cast(pl.Utf8))
            d_dep = d_dep.with_columns(pl.col(key_dep).cast(pl.Utf8))
            if desc_dep and desc_dep in d_dep.columns: d_dep = d_dep.rename({desc_dep: "DEP_DESC"})
            res = res.join(d_dep, left_on="Depósito", right_on=key_dep, how="left", suffix="_dep_dim")

    # 2. Estatísticas
    daily_agg = (
        res.filter(pl.col("Data").is_not_null())
        .group_by(["Depósito", "SKU", "Data"])
        .agg(pl.col("Quantidade").sum().alias("Qtd_Dia"))
    )
    
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
    
    # 3. Traz Descrições
    if "SKU_DESC" in res.columns:
        desc_s = res.group_by("SKU").agg(pl.col("SKU_DESC").first())
        stats = stats.join(desc_s, on="SKU", how="left")
    
    if "DEP_DESC" in res.columns:
        desc_d = res.group_by("Depósito").agg(pl.col("DEP_DESC").first())
        stats = stats.join(desc_d, on="Depósito", how="left")

    # 4. Padronização Final
    stats = stats.rename({"Depósito": "Código Depósito", "SKU": "Código SKU"})
    
    if "SKU_DESC" in stats.columns: stats = stats.rename({"SKU_DESC": "SKU"})
    else: stats = stats.with_columns(pl.lit("").alias("SKU"))
        
    if "DEP_DESC" in stats.columns: stats = stats.rename({"DEP_DESC": "Depósito"})
    else: stats = stats.with_columns(pl.lit("").alias("Depósito"))

    final_cols = [
        "Código Depósito", "Depósito", 
        "Código SKU", "SKU", 
        "Média", "Máximo", "Desvio", 
        "Média + 1 Desv", "Média + 2 Desv", "Média + 3 Desv", 
        "Percentil 95%"
    ]
    
    ignore = set(final_cols + ["Qtd_Dia", "Data", "Hora", "Pedido", "Caixa", "Rota/Destino"])
    extra_cols = [c for c in stats.columns if c not in ignore]
    
    return stats.select(final_cols + extra_cols), res

def get_bytes(df: pl.DataFrame, fmt: str) -> bytes:
    b = io.BytesIO()
    if fmt == 'xlsx': df.write_excel(b)
    elif fmt == 'csv': df.write_csv(b)
    return b.getvalue()

# ==============================================================================
# 3. UI PRINCIPAL (WIZARD FLOW)
# ==============================================================================

def main():
    setup_page()

    # --- CABEÇALHO COM LOGOTIPO ---
    c_logo, c_title, c_act = st.columns([0.15, 0.65, 0.2], vertical_alignment="bottom")
    
    with c_logo:
        st.image("Aguia Fundo Branco.png", use_container_width=True)
        
    with c_title:
        st.markdown("""
            <h3 style='margin: 0; padding-bottom: 35px; font-weight: 600; color: #1e293b;'>
                Design Soluções | Movimentações Clientes
            </h3>
        """, unsafe_allow_html=True)
        
    with c_act:
        if st.button("🔄 Novo Projeto", type="secondary"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    
    st.markdown("---")

    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
        st.session_state.mapping = {}
        st.session_state.split_dt = True
        st.session_state.dt_source = None

    # ==========================================================================
    # ETAPA 1
    # ==========================================================================
    if st.session_state.current_step > 1:
        st.markdown("""
            <div class="step-summary">
                <div class="step-check">✓</div>
                <div class="step-text">Etapa 1: Configuração Concluída.</div>
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.current_step == 1:
        st.markdown("""
            <div class="step-header-card">
                <span class="step-badge">ETAPA 1</span>
                <h3 class="step-title">Configuração Inicial</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("Carregue uma amostra para identificar a estrutura dos dados.")
        f_sample = st.file_uploader("Arquivo de Amostra", type=["xlsx", "csv"], label_visibility="collapsed")
        if f_sample:
            df_s = load_sample(f_sample)
            st.session_state.cols_origem = ["--- Ignorar ---"] + df_s.columns
            st.session_state.current_step = 2
            st.rerun()

    # ==========================================================================
    # ETAPA 2
    # ==========================================================================
    if st.session_state.current_step > 2:
        st.markdown("""
            <div class="step-summary">
                <div class="step-check">✓</div>
                <div class="step-text">Etapa 2: Mapeamento Definido.</div>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.current_step == 2:
        st.markdown("""
            <div class="step-header-card">
                <span class="step-badge">ETAPA 2</span>
                <h3 class="step-title">Mapeamento de Colunas</h3>
            </div>
        """, unsafe_allow_html=True)
        
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
        if st.button("Salvar e Avançar", type="primary"):
            st.session_state.mapping = curr_map
            st.session_state.split_dt = split_dt
            st.session_state.dt_source = dt_source
            st.session_state.current_step = 3
            st.rerun()

    # ==========================================================================
    # ETAPA 3
    # ==========================================================================
    if st.session_state.current_step > 3:
        st.markdown("""
            <div class="step-summary">
                <div class="step-check">✓</div>
                <div class="step-text">Etapa 3: Dados Processados e Enriquecidos.</div>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.current_step == 3:
        st.markdown("""
            <div class="step-header-card">
                <span class="step-badge">ETAPA 3</span>
                <h3 class="step-title">Processamento em Lote</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col_main, col_dim = st.columns([1, 1])
        with col_main:
            st.markdown("##### 1. Movimentação (Lote)")
            files_mov = st.file_uploader("Arquivos Movimentação", type=["xlsx", "csv"], accept_multiple_files=True)
        with col_dim:
            st.markdown("##### 2. Dimensões (Opcional)")
            tab_s, tab_d = st.tabs(["📦 SKU", "🏢 Depósito"])
            with tab_s:
                f_sku = st.file_uploader("Dimensão SKU", type=["xlsx", "csv"])
                k_sku, d_sku = None, None
                if f_sku:
                    df_pre = load_sample(f_sku)
                    k_sku = st.selectbox("Chave Código:", df_pre.columns, key="ks")
                    d_sku = st.selectbox("Col. Descrição:", ["---"] + df_pre.columns, key="ds")
            with tab_d:
                f_dep = st.file_uploader("Dimensão Depósito", type=["xlsx", "csv"])
                k_dep, d_dep = None, None
                if f_dep:
                    df_pre = load_sample(f_dep)
                    k_dep = st.selectbox("Chave Código:", df_pre.columns, key="kd")
                    d_dep = st.selectbox("Col. Descrição:", ["---"] + df_pre.columns, key="dd")
        st.markdown("###")
        if files_mov:
            if st.button("🚀 Processar Dados", type="primary"):
                main_df = process_etl_batch(files_mov, st.session_state.mapping, st.session_state.split_dt, st.session_state.dt_source)
                if not main_df.is_empty():
                    with st.status("Processando...", expanded=True):
                        st.write("Consolidando...")
                        stats_df, detail_df = enrich_and_calculate_stats(main_df, f_sku, k_sku, d_sku, f_dep, k_dep, d_dep)
                        st.session_state.final_stats = stats_df
                        st.session_state.detail_df = detail_df
                    st.session_state.current_step = 4
                    st.rerun()

    # ==========================================================================
    # ETAPA 4: ANÁLISE INTERATIVA
    # ==========================================================================
    if st.session_state.current_step > 4:
        st.markdown("""
            <div class="step-summary">
                <div class="step-check">✓</div>
                <div class="step-text">Etapa 4: Análise Visual Concluída.</div>
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.current_step == 4:
        stats = st.session_state.final_stats.clone()
        detail = st.session_state.detail_df.clone()

        st.markdown("""
            <div class="step-header-card">
                <span class="step-badge">ETAPA 4</span>
                <h3 class="step-title">Dashboard de Análise</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Filtros Globais
        stats = stats.with_columns([
            pl.concat_str([pl.col("Código SKU"), pl.lit(" - "), pl.col("SKU").fill_null("")]).alias("Label_SKU"),
            pl.concat_str([pl.col("Código Depósito"), pl.lit(" - "), pl.col("Depósito").fill_null("")]).alias("Label_Dep")
        ])
        
        c_f1, c_f2 = st.columns(2)
        with c_f1: sel_skus = st.multiselect("Filtrar SKUs:", stats["Label_SKU"].unique().sort().to_list())
        with c_f2: sel_deps = st.multiselect("Filtrar Depósitos:", stats["Label_Dep"].unique().sort().to_list())
        
        v_stats, v_detail = stats, detail
        
        if sel_skus:
            codes_s = [s.split(" - ")[0] for s in sel_skus]
            v_stats = v_stats.filter(pl.col("Label_SKU").is_in(sel_skus))
            v_detail = v_detail.filter(pl.col("SKU").cast(pl.Utf8).is_in(codes_s))
            
        if sel_deps:
            codes_d = [d.split(" - ")[0] for d in sel_deps]
            v_stats = v_stats.filter(pl.col("Label_Dep").is_in(sel_deps))
            v_detail = v_detail.filter(pl.col("Depósito").cast(pl.Utf8).is_in(codes_d))

        # --- SELEÇÃO NA TABELA (DRILL DOWN) ---
        st.markdown("###")
        st.markdown("**Selecione uma linha na tabela para filtrar o dashboard:**")
        
        if 'selected_row' in st.session_state:
            if st.button("❌ Limpar Seleção de Linha", type="secondary"):
                del st.session_state.selected_row
                st.rerun()

        pdf_display = v_stats.drop(["Label_SKU", "Label_Dep"]).to_pandas()
        
        selection = st.dataframe(
            pdf_display, 
            use_container_width=True, 
            height=350,
            column_config={"Média": st.column_config.NumberColumn(format="%.2f"), "Desvio": st.column_config.NumberColumn(format="%.2f")},
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if selection.selection.rows:
            idx = selection.selection.rows[0]
            row_data = pdf_display.iloc[idx]
            sel_sku_code = str(row_data["Código SKU"])
            sel_dep_code = str(row_data["Código Depósito"])
            
            st.session_state.selected_row = f"{sel_sku_code} | {sel_dep_code}"
            
            v_detail = v_detail.filter(
                (pl.col("SKU").cast(pl.Utf8) == sel_sku_code) & 
                (pl.col("Depósito").cast(pl.Utf8) == sel_dep_code)
            )
            st.info(f"🔎 Filtrando dashboard para SKU: {sel_sku_code} no Depósito: {sel_dep_code}")

        # --- BIG NUMBERS (KPIs) ---
        st.markdown("###")
        if v_detail.height > 0:
            qtd_linhas = v_detail.height
            
            # SAFE KPI Calculation (prevents NoneType error)
            qtd_unidades = v_detail["Quantidade"].sum()
            qtd_unidades = qtd_unidades if qtd_unidades is not None else 0
            
            qtd_picking = v_detail["Pedido"].n_unique() if "Pedido" in v_detail.columns else 0
            qtd_skus = v_detail["SKU"].n_unique()
            qtd_deps = v_detail["Depósito"].n_unique()
            
            daily_agg = v_detail.group_by("Data").agg(pl.col("Quantidade").sum()).sort("Data")
            qtd_dias = daily_agg.height
            
            avg_day = daily_agg["Quantidade"].mean()
            avg_day = avg_day if avg_day is not None else 0
            
            max_day = daily_agg["Quantidade"].max()
            max_day = max_day if max_day is not None else 0

            # LINHA 1
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Qtde. de códigos de picking</div><div class="kpi-value">{qtd_picking:,}</div></div>""".replace(",", "."), unsafe_allow_html=True)
            k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Qtde. de unidades na base</div><div class="kpi-value">{qtd_unidades:,.0f}</div></div>""".replace(",", "."), unsafe_allow_html=True)
            k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Qtde. de linhas na base</div><div class="kpi-value">{qtd_linhas:,}</div></div>""".replace(",", "."), unsafe_allow_html=True)
            k4.markdown(f"""<div class="kpi-card"><div class="kpi-label">Qtde. de SKUs na base</div><div class="kpi-value">{qtd_skus:,}</div></div>""".replace(",", "."), unsafe_allow_html=True)
            k5.markdown(f"""<div class="kpi-card"><div class="kpi-label">Dias de informação</div><div class="kpi-value">{qtd_dias}</div></div>""", unsafe_allow_html=True)
            
            st.markdown("###")
            # LINHA 2
            kt1, kt2, kt3 = st.columns(3)
            kt1.markdown(f"""<div class="kpi-card"><div class="kpi-label">Depósitos Únicos</div><div class="kpi-value">{qtd_deps}</div><div class="kpi-sub">Locais de Estoque</div></div>""", unsafe_allow_html=True)
            kt2.markdown(f"""<div class="kpi-card"><div class="kpi-label">Média Diária</div><div class="kpi-value">{avg_day:,.0f}</div><div class="kpi-sub">Unidades / Dia</div></div>""".replace(",", "."), unsafe_allow_html=True)
            kt3.markdown(f"""<div class="kpi-card"><div class="kpi-label">Pico Máximo</div><div class="kpi-value">{max_day:,.0f}</div><div class="kpi-sub">Recorde em um dia</div></div>""".replace(",", "."), unsafe_allow_html=True)

        # --- GRÁFICOS ---
        if v_detail.height > 0:
            st.markdown("---")
            
            # 1. Gráfico de COLUNAS
            daily_trend = daily_agg.to_pandas()
            fig_trend = px.bar(
                daily_trend, x="Data", y="Quantidade",
                title="Volume Diário de Movimentação",
                color_discrete_sequence=["#2dd4bf"]
            )
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title=None, yaxis_gridcolor="#e2e8f0"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # 2. HEATMAP "GITHUB STYLE"
            hm_data = (
                v_detail.filter(pl.col("Data").is_not_null())
                .group_by("Data")
                .agg(pl.col("Quantidade").sum().alias("Qtd"))
                .sort("Data")
                .with_columns([
                    pl.col("Data").dt.strftime("%Y-W%U").alias("YearWeek"),
                    pl.col("Data").dt.strftime("%a").alias("DiaSemana")
                ])
                .to_pandas()
            )
            
            fig_hm = px.density_heatmap(
                hm_data, x="YearWeek", y="DiaSemana", z="Qtd",
                color_continuous_scale="Greens",
                title="Intensidade de Atividade (Semanal)",
                category_orders={
                    "DiaSemana": ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]
                }
            )
            fig_hm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Semana", yaxis_title=None,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig_hm.update_traces(xgap=3, ygap=3)
            st.plotly_chart(fig_hm, use_container_width=True)
        
        st.markdown("###")
        if st.button("Ir para Exportação", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

    # ==========================================================================
    # ETAPA 5: EXPORTAÇÃO
    # ==========================================================================
    if st.session_state.current_step == 5:
        stats_final = st.session_state.final_stats
        cols_to_drop = [c for c in ["Label_SKU", "Label_Dep"] if c in stats_final.columns]
        if cols_to_drop: stats_final = stats_final.drop(cols_to_drop)
        
        st.markdown("""
            <div class="step-header-card">
                <span class="step-badge">ETAPA 5</span>
                <h3 class="step-title">Downloads</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.success("Processo concluído com sucesso.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📄 Tabela Analítica")
            b_xls = get_bytes(stats_final, 'xlsx')
            st.download_button("Baixar Excel (.xlsx)", b_xls, "analise_sku.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with c2:
            st.markdown("##### 📦 Base Detalhada")
            if st.button("Gerar Arquivo Completo (CSV)", use_container_width=True):
                with st.spinner("Gerando..."):
                    b_det = get_bytes(st.session_state.detail_df, 'csv')
                    st.session_state.b_det = b_det
            if 'b_det' in st.session_state:
                st.download_button("📥 Baixar CSV Completo", st.session_state.b_det, "base_completa.csv", "text/csv", use_container_width=True)

        st.markdown("---")
        if st.button("⬅️ Voltar para Análise"):
            st.session_state.current_step = 4
            st.rerun()

if __name__ == "__main__":
    main()
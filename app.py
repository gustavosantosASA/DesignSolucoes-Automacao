import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import time
import gc
import os
import warnings

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO INICIAL
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        
        .kpi-card { 
            background-color: #ffffff; 
            border: 1px solid #e2e8f0; 
            border-radius: 12px; 
            padding: 15px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.02); 
            text-align: left; 
            height: 100%; 
            display: flex; 
            flex-direction: column; 
            justify-content: center;
            cursor: help;
        }
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
# 2. MOTOR DE DADOS
# ==============================================================================

def read_file_chunk(file) -> pl.DataFrame:
    try:
        if file.name.endswith('.csv'):
            return pl.read_csv(file, ignore_errors=True, infer_schema_length=0)
        else:
            try: return pl.read_excel(file, engine="calamine")
            except:
                try: return pl.read_excel(file) 
                except:
                    file.seek(0)
                    return pl.from_pandas(pd.read_excel(file))
    except Exception: return pl.DataFrame()

def load_dim_safe(file):
    if not file: return pl.DataFrame()
    df = read_file_chunk(file)
    if not df.is_empty():
        return df.select([pl.col(c).cast(pl.Utf8) for c in df.columns])
    return df

def process_and_clean_single_file(df_raw, mapping, split_dt, dt_source):
    if df_raw.is_empty(): return None
    exprs = []
    
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

    target_cols = ["Depósito", "SKU", "Pedido", "Caixa", "Quantidade", "Rota/Destino"]
    for target in target_cols:
        src = mapping.get(target)
        if src and src in df_raw.columns:
            if target == "Quantidade":
                exprs.append(
                    pl.col(src).cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0).cast(pl.Float32).alias(target)
                )
            else:
                exprs.append(pl.col(src).cast(pl.Utf8, strict=False).fill_null("").alias(target))
        else:
            if target == "Quantidade":
                exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(target))
            else:
                exprs.append(pl.lit("", dtype=pl.Utf8).alias(target))

    return df_raw.select(exprs)

def load_sample_optimized(file) -> pl.DataFrame:
    try:
        if file.name.endswith('.csv'): 
            return pl.read_csv(file, n_rows=100, ignore_errors=True, try_parse_dates=True)
        else: 
            try: return pl.read_excel(file, engine="calamine")
            except: return pl.read_excel(file) 
    except: return pl.DataFrame()

def enrich_and_calculate_stats(main_df, dim_sku_file, key_sku, desc_sku, dim_dep_file, key_dep, desc_dep):
    res = main_df
    res = res.with_columns([pl.col("SKU").cast(pl.Utf8).fill_null(""), pl.col("Depósito").cast(pl.Utf8).fill_null("")])

    if dim_sku_file and key_sku:
        d_sku = load_dim_safe(dim_sku_file)
        if not d_sku.is_empty() and key_sku in d_sku.columns:
            d_sku = d_sku.with_columns(pl.col(key_sku).cast(pl.Utf8))
            if desc_sku and desc_sku in d_sku.columns: d_sku = d_sku.rename({desc_sku: "SKU_DESC"})
            res = res.join(d_sku, left_on="SKU", right_on=key_sku, how="left", suffix="_sku_dim")

    if dim_dep_file and key_dep:
        d_dep = load_dim_safe(dim_dep_file)
        if not d_dep.is_empty() and key_dep in d_dep.columns:
            d_dep = d_dep.with_columns(pl.col(key_dep).cast(pl.Utf8))
            if desc_dep and desc_dep in d_dep.columns: d_dep = d_dep.rename({desc_dep: "DEP_DESC"})
            res = res.join(d_dep, left_on="Depósito", right_on=key_dep, how="left", suffix="_dep_dim")

    daily_agg = res.filter(pl.col("Data").is_not_null()).group_by(["Depósito", "SKU", "Data"]).agg(pl.col("Quantidade").sum().alias("Qtd_Dia"))
    
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
    
    if "SKU_DESC" in res.columns:
        desc_s = res.group_by("SKU").agg(pl.col("SKU_DESC").first())
        stats = stats.join(desc_s, on="SKU", how="left")
    else:
        stats = stats.with_columns(pl.lit("-").alias("SKU_DESC"))
        
    if "DEP_DESC" in res.columns:
        desc_d = res.group_by("Depósito").agg(pl.col("DEP_DESC").first())
        stats = stats.join(desc_d, on="Depósito", how="left")
    else:
        stats = stats.with_columns(pl.lit("-").alias("DEP_DESC"))

    # RENOMEAÇÃO FINAL
    stats = stats.rename({
        "Depósito": "Código Depósito", 
        "DEP_DESC": "Depósito",
        "SKU": "SKU", # Mantém SKU como SKU, não Código SKU
        "SKU_DESC": "Descrição"
    })

    final_cols = [
        "Código Depósito", "Depósito", "SKU", "Descrição",
        "Média", "Máximo", "Desvio", 
        "Média + 1 Desv", "Média + 2 Desv", "Média + 3 Desv", 
        "Percentil 95%"
    ]
    
    for c in final_cols:
        if c not in stats.columns:
            stats = stats.with_columns(pl.lit("-").alias(c))

    return stats.select(final_cols), res

def get_bytes(df: pl.DataFrame, fmt: str) -> bytes:
    b = io.BytesIO()
    if fmt == 'xlsx': df.write_excel(b)
    elif fmt == 'csv': df.write_csv(b)
    return b.getvalue()

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
            for k in list(st.session_state.keys()): del st.session_state[k]
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
            df_s = load_sample_optimized(f_sample)
            st.session_state.cols_origem = ["--- Ignorar ---"] + df_s.columns
            st.session_state.current_step = 2
            st.rerun()

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
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 3: Dados Processados e Enriquecidos.</div></div>""", unsafe_allow_html=True)
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
                k_sku, d_sku = None, None
                if f_sku:
                    df_pre = load_sample_optimized(f_sku)
                    k_sku = st.selectbox("Chave Código:", df_pre.columns, key="ks")
                    d_sku = st.selectbox("Col. Descrição:", ["---"] + df_pre.columns, key="ds")
            with tab_d:
                f_dep = st.file_uploader("Dimensão Depósito", type=["xlsx", "csv"])
                k_dep, d_dep = None, None
                if f_dep:
                    df_pre = load_sample_optimized(f_dep)
                    k_dep = st.selectbox("Chave Código:", df_pre.columns, key="kd")
                    d_dep = st.selectbox("Col. Descrição:", ["---"] + df_pre.columns, key="dd")
        
        st.markdown("###")
        if files_mov:
            if st.button("🚀 Processar Dados", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Iniciando motor de dados...")
                dfs_list = []
                total_files = len(files_mov)
                for idx, file in enumerate(files_mov):
                    try:
                        progress_bar.progress((idx) / total_files, text=f"Processando: {file.name}")
                        df_raw = read_file_chunk(file)
                        df_clean = process_and_clean_single_file(
                            df_raw, st.session_state.mapping, st.session_state.split_dt, st.session_state.dt_source
                        )
                        if df_clean is not None: dfs_list.append(df_clean)
                        del df_raw
                        gc.collect()
                    except Exception as e:
                        st.error(f"Erro no arquivo {file.name}: {str(e)}")
                
                progress_bar.progress(0.9, text="Consolidando dados...")
                
                if dfs_list:
                    main_df = pl.concat(dfs_list, how="vertical_relaxed")
                    del dfs_list
                    gc.collect()
                    
                    stats_df, detail_df = enrich_and_calculate_stats(main_df, f_sku, k_sku, d_sku, f_dep, k_dep, d_dep)
                    st.session_state.final_stats = stats_df
                    st.session_state.detail_df = detail_df
                    st.session_state.current_step = 4
                    st.rerun()
                else:
                    st.error("Nenhum dado válido foi processado.")

    # --- ETAPA 4 ---
    if st.session_state.current_step > 4:
        st.markdown("""<div class="step-summary"><div class="step-check">✓</div><div class="step-text">Etapa 4: Análise Visual Concluída.</div></div>""", unsafe_allow_html=True)
    if st.session_state.current_step == 4:
        stats = st.session_state.final_stats.clone()
        detail = st.session_state.detail_df.clone()

        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 4</span><h3 class="step-title">Dashboard de Análise</h3></div>""", unsafe_allow_html=True)
        
        # Correção aqui: Uso de "SKU" e "Descrição" conforme novos nomes
        stats = stats.with_columns([
            pl.concat_str([pl.col("SKU"), pl.lit(" - "), pl.col("Descrição")]).alias("Label_SKU"),
            pl.concat_str([pl.col("Código Depósito"), pl.lit(" - "), pl.col("Depósito")]).alias("Label_Dep")
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

        if v_detail.height > 0:
            qtd_linhas = v_detail.height
            qtd_unidades = v_detail["Quantidade"].sum()
            qtd_picking = v_detail["Pedido"].n_unique() if "Pedido" in v_detail.columns else 0
            qtd_skus = v_detail["SKU"].n_unique()
            qtd_deps = v_detail["Depósito"].n_unique()
            
            daily_agg = v_detail.group_by("Data").agg(pl.col("Quantidade").sum()).sort("Data")
            qtd_dias = daily_agg.height
            avg_day = daily_agg["Quantidade"].mean() or 0
            max_day = daily_agg["Quantidade"].max() or 0

            def kpi_html(label, value, sub, tooltip):
                return f"""
                <div class="kpi-card" title="{tooltip}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(kpi_html("Linhas", f"{qtd_linhas:,}".replace(",", "."), "Registros Totais", "Contagem total de linhas."), unsafe_allow_html=True)
            k2.markdown(kpi_html("Volume (Unid)", f"{qtd_unidades:,.0f}".replace(",", "."), "Soma Total", "Soma absoluta da coluna Quantidade."), unsafe_allow_html=True)
            k3.markdown(kpi_html("Picking", f"{qtd_picking:,}".replace(",", "."), "Pedidos Únicos", "Contagem distinta de pedidos."), unsafe_allow_html=True)
            k4.markdown(kpi_html("SKUs", f"{qtd_skus:,}".replace(",", "."), "Produtos Únicos", "Contagem distinta de SKUs."), unsafe_allow_html=True)
            k5.markdown(kpi_html("Dias", f"{qtd_dias}", "Dias com Movimento", "Dias únicos na base."), unsafe_allow_html=True)
            
            st.markdown("###")
            kt1, kt2, kt3 = st.columns(3)
            kt1.markdown(kpi_html("Depósitos", f"{qtd_deps}", "Locais Únicos", "Contagem distinta de Depósitos."), unsafe_allow_html=True)
            kt2.markdown(kpi_html("Média Diária", f"{avg_day:,.0f}".replace(",", "."), "Unidades / Dia", "Volume Total / Dias com movimento."), unsafe_allow_html=True)
            kt3.markdown(kpi_html("Pico Máximo", f"{max_day:,.0f}".replace(",", "."), "Recorde em 1 Dia", "Maior volume em um único dia."), unsafe_allow_html=True)
            
            st.caption("ℹ️ Passe o mouse sobre os cards para ver o significado.")

            st.markdown("---")
            daily_trend = daily_agg.to_pandas()
            fig_trend = px.bar(daily_trend, x="Data", y="Quantidade", title="Volume Diário", color_discrete_sequence=["#2dd4bf"], template="plotly_white")
            fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, font_color="#334155")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            real_data_pdf = daily_agg.to_pandas()
            real_data_pdf["Data"] = pd.to_datetime(real_data_pdf["Data"])
            if not real_data_pdf.empty:
                min_date = real_data_pdf["Data"].min()
                date_range = pd.date_range(start=min_date, periods=54 * 7, freq='D')
                skeleton_df = pd.DataFrame({"Data": date_range})
                hm_final = pd.merge(skeleton_df, real_data_pdf, on="Data", how="left").fillna(0)
                hm_final["YearWeek"] = hm_final["Data"].dt.strftime("%Y-W%U")
                hm_final["DiaSemana"] = hm_final["Data"].dt.strftime("%a")
                
                fig_hm = px.density_heatmap(
                    hm_final, x="YearWeek", y="DiaSemana", z="Quantidade", # Corrigido para 'Quantidade' pois vem do daily_agg
                    color_continuous_scale="Greens", title="Intensidade (54 Semanas)",
                    category_orders={"DiaSemana": ["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"]},
                    range_color=[0, hm_final["Quantidade"].max()], template="plotly_white"
                )
                fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, margin=dict(l=20, r=20, t=40, b=20), font_color="#334155")
                fig_hm.update_traces(xgap=3, ygap=3, showscale=True)
                st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("###")
        st.markdown("**Tabela Analítica (Selecione para filtrar):**")
        
        if 'selected_row' in st.session_state:
            if st.button("❌ Limpar Seleção", type="secondary"):
                del st.session_state.selected_row
                st.rerun()

        pdf_display = v_stats.drop(["Label_SKU", "Label_Dep"]).to_pandas()
        selection = st.dataframe(
            pdf_display, 
            use_container_width=True, 
            height=350,
            column_config={
                "Média": st.column_config.NumberColumn(format="%.2f"), 
                "Desvio": st.column_config.NumberColumn(format="%.2f")
            },
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if selection.selection.rows:
            idx = selection.selection.rows[0]
            row_data = pdf_display.iloc[idx]
            # Correção para usar os novos nomes de coluna
            sel_sku_code = str(row_data["SKU"])
            sel_dep_code = str(row_data["Código Depósito"])
            st.session_state.selected_row = f"{sel_sku_code}|{sel_dep_code}"
            
            st.rerun()

        st.markdown("###")
        if st.button("Ir para Exportação", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

    # --- ETAPA 5 ---
    if st.session_state.current_step == 5:
        stats_final = st.session_state.final_stats
        cols_to_drop = [c for c in ["Label_SKU", "Label_Dep"] if c in stats_final.columns]
        if cols_to_drop: stats_final = stats_final.drop(cols_to_drop)
        st.markdown("""<div class="step-header-card"><span class="step-badge">ETAPA 5</span><h3 class="step-title">Downloads</h3></div>""", unsafe_allow_html=True)
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
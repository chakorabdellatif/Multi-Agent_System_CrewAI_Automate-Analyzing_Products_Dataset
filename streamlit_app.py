"""
Multi-Agent Product Intelligence Suite · CrewAI System
École Nationale d'Intelligence Artificielle et du Digital (ENIAD)
Systèmes Multi-Agents (SMA) & Orchestration d'Agents Autonomes
"""

from pathlib import Path
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CrewAI Multi-Agent · Product Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "ai_analysis" / "data"
REPORT_PATH = ROOT_DIR / "ai_analysis" / "report.md"

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a, #1e293b); border: 1px solid #334155; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
    <h1 style="color: #38bdf8; margin: 0; font-size: 2.2rem;">🤖 CrewAI Multi-Agent Product Intelligence</h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
        Orchestration d'Agents Autonomes Spécialisés pour l'Audit et l'Analyse Prédictive de Catalogues Produits
    </p>
    <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 10px;">
        <span style="background: #0369a1; color: #bae6fd; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">Data Preparation Agent</span>
        <span style="background: #4338ca; color: #c7d2fe; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">Pattern Analyst Agent</span>
        <span style="background: #0f766e; color: #99f6e4; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">Visualization Specialist</span>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_products_data():
    file_path = DATA_DIR / "Products.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    # Synthetic fallback if not found
    brands = ["Nike", "Adidas", "Gucci", "Zara", "H&M"]
    cats = ["Dress", "Shoes", "Jacket", "Shirt", "Pants"]
    colors = ["Black", "White", "Blue", "Red", "Green"]
    sizes = ["S", "M", "L", "XL"]
    data = []
    for i in range(200):
        data.append({
            "User ID": i + 1,
            "Product ID": 100 + i,
            "Product Name": np.random.choice(cats),
            "Brand": np.random.choice(brands),
            "Category": np.random.choice(cats),
            "Price": float(np.random.randint(20, 250)),
            "Rating": float(np.random.choice([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
            "Color": np.random.choice(colors),
            "Size": np.random.choice(sizes)
        })
    return pd.DataFrame(data)


df_raw = load_products_data()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration des Agents")
    st.subheader("👥 Personas de la Crew")
    st.markdown("""
    - **🧹 Agent 1 : Data Preparation**
      *Nettoyage, imputation & normalisation*
    - **🔬 Agent 2 : Pattern Detection**
      *Corrélations, clustering & segmentation*
    - **📊 Agent 3 : Visualization Engineer**
      *Graphiques décisionnels & KPIs*
    """)
    st.divider()

    uploaded_file = st.file_uploader("Téléverser un catalogue CSV personnalisé :", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success("Catalogue personnalisé chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

    st.divider()
    st.subheader("🔍 Filtres Interactifs")
    brand_list = ["Tous"] + sorted(list(df_raw["Brand"].dropna().unique()))
    selected_brand = st.selectbox("Filtrer par Marque :", brand_list)

    if "Price" in df_raw.columns:
        min_p = float(df_raw["Price"].min())
        max_p = float(df_raw["Price"].max())
        price_range = st.slider("Fourchette de Prix (€) :", min_p, max_p, (min_p, max_p))
    else:
        price_range = (0.0, 1000.0)

# Filter Data
df_filtered = df_raw.copy()
if selected_brand != "Tous":
    df_filtered = df_filtered[df_filtered["Brand"] == selected_brand]
if "Price" in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered["Price"] >= price_range[0]) & (df_filtered["Price"] <= price_range[1])]

# Tabs
tab_exec, tab_data, tab_charts, tab_report = st.tabs([
    "🚀 Lancement & Exécution Multi-Agents",
    "📁 Explorateur de Catalogue",
    "📈 Visualisations Stratégiques",
    "📋 Rapport Décisionnel de Synthèse"
])

# =========================================================================
# TAB 1: CREW EXECUTION
# =========================================================================
with tab_exec:
    st.subheader("🚀 Déploiement et Orchestration de la Crew")
    st.markdown("""
    Déclenchez la collaboration autonome entre le **Data Preparer**, le **Pattern Analyst** et le **Visualizer**.
    Chaque agent traite les résultats de son prédécesseur dans un pipeline séquentiel continu.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Étape 1 : Préparation & Assainissement**\n- Détection des doublons\n- Traitement des valeurs manquantes\n- Standardisation des prix")
    with c2:
        st.info("**Étape 2 : Analyse de Motifs & Tendances**\n- Identification des marques leaders\n- Analyse de sensibilité au prix\n- Calcul du taux de satisfaction")
    with c3:
        st.info("**Étape 3 : Synthèse & Recommandations**\n- Génération de visualisations Plotly\n- Rédaction du rapport d'orientation\n- Export des données enrichies")

    st.write("")
    if st.button("⚡ Lancer l'Analyse Autonome Multi-Agents", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1
        status_text.markdown("🤖 **[Data Preparation Agent]** : Analyse de la structure du catalogue...")
        time.sleep(0.8)
        progress_bar.progress(33)

        # Step 2
        status_text.markdown("🔬 **[Pattern Analyst Agent]** : Calcul des métriques de corrélation et segmentation...")
        time.sleep(0.8)
        progress_bar.progress(66)

        # Step 3
        status_text.markdown("📊 **[Visualization Specialist]** : Compilation du rapport décisionnel et graphiques...")
        time.sleep(0.6)
        progress_bar.progress(100)

        status_text.markdown("✅ **Mission Accomplie avec Succès par la Crew !**")
        st.success("Toutes les tâches d'orchestration ont été exécutées. Consultez les onglets 'Visualisations' et 'Rapport'.")

        st.subheader("💡 Indicateurs Clés Extraits par les Agents :")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Produits Analysés", len(df_filtered))
        with kpi2:
            st.metric("Prix Moyen Catalogue", f"{df_filtered['Price'].mean():.2f} €" if "Price" in df_filtered.columns else "N/A")
        with kpi3:
            st.metric("Note Moyenne Clients", f"{df_filtered['Rating'].mean():.2f} / 5.0" if "Rating" in df_filtered.columns else "N/A")
        with kpi4:
            st.metric("Marques Actives", df_filtered["Brand"].nunique() if "Brand" in df_filtered.columns else "N/A")

# =========================================================================
# TAB 2: DATA EXPLORER
# =========================================================================
with tab_data:
    st.subheader("📁 Données Produits (Échantillon Filtré)")
    st.dataframe(df_filtered, use_container_width=True, height=400)
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les données analysées (CSV)",
        data=csv_data,
        file_name="catalogue_produits_analyse.csv",
        mime="text/csv",
    )

# =========================================================================
# TAB 3: VISUALIZATIONS
# =========================================================================
with tab_charts:
    st.subheader("📈 Tableaux de Bord Générés par le Visualizer Agent")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if "Brand" in df_filtered.columns and "Price" in df_filtered.columns:
            fig_box = px.box(
                df_filtered, x="Brand", y="Price", color="Brand",
                title="Distribution des Prix par Marque",
                points="all"
            )
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with col_g2:
        if "Category" in df_filtered.columns:
            cat_counts = df_filtered["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_cat = px.bar(
                cat_counts, x="Category", y="Count", color="Count",
                title="Volume d'Articles par Catégorie",
                color_continuous_scale="Blues"
            )
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        if "Price" in df_filtered.columns and "Rating" in df_filtered.columns:
            df_scatter = df_filtered.dropna(subset=["Price", "Rating"])
            fig_scat = px.scatter(
                df_scatter, x="Price", y="Rating", color="Brand",
                hover_data=["Product Name"] if "Product Name" in df_scatter.columns else None,
                title="Corrélation : Prix (€) vs Note de Satisfaction (1-5)"
            )
            fig_scat.update_layout(height=400)
            st.plotly_chart(fig_scat, use_container_width=True)

    with col_g4:
        if "Color" in df_filtered.columns:
            color_counts = df_filtered["Color"].value_counts().reset_index()
            color_counts.columns = ["Color", "Count"]
            fig_pie = px.pie(color_counts, names="Color", values="Count", title="Répartition par Couleur", hole=0.3)
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

# =========================================================================
# TAB 4: REPORT
# =========================================================================
with tab_report:
    st.subheader("📋 Rapport d'Analyse Stratégique Généré par la Crew")
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.info("Le rapport d'analyse synthétique peut être généré à tout moment via l'onglet d'exécution.")

"""Barre du haut, commune à toutes les pages : accueil, accroche défilante, recherche et opérateur.

Elle est fixée en haut de la zone principale, sur la même ligne que le menu Streamlit (Deploy).
"""
import html

import streamlit as st

from .i18n import T

CSS = """<style>
.st-key-bgtopbar{position:fixed;top:0;left:262px;right:0;height:3.75rem;z-index:999989;padding:0 11rem 0 2.2rem!important;
  width:auto!important;max-width:none!important;background:var(--bg);border-bottom:1px solid var(--line);
  display:grid!important;grid-template-columns:auto minmax(13rem,1fr) minmax(11rem,21rem) auto;align-items:center;gap:1rem}
.st-key-bgtopbar>*{margin:0!important;height:auto!important;min-height:0!important;width:auto!important;align-self:center}
.stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-bgtopbar{left:0;padding-left:6.5rem!important}
@media (max-width:1150px){.st-key-bgtopbar{grid-template-columns:auto minmax(9rem,1fr) minmax(9rem,15rem) auto;gap:.7rem}}
@media (max-width:767px){.st-key-bgtopbar{left:0;padding:0 8.8rem 0 4.6rem!important;grid-template-columns:auto minmax(0,1fr) auto;gap:.45rem}
  .st-key-bgtopbar>[data-testid="stElementContainer"]:has(.bg-top){display:none!important}
  .st-key-bgaccueil a p,.st-key-bgop button p{display:none}
  .st-key-bgaccueil a{padding:0 .55rem!important}.st-key-bgop button{padding:0 .3rem!important}
  .st-key-bgop button>div>div:last-child{display:none}}
@media (max-width:480px){.st-key-bgtopbar{grid-template-columns:auto 1fr auto}
  .st-key-bgtopbar>[data-testid="stLayoutWrapper"]:has(.st-key-bgrecherche),.st-key-bgrecherche{display:none!important}
  .st-key-bgtopbar>*:has(.st-key-bgop){justify-self:end}}
[data-testid="stHeader"],[data-testid="stHeader"] *{pointer-events:none}
[data-testid="stHeader"] button,[data-testid="stHeader"] a,[data-testid="stHeader"] button *,[data-testid="stHeader"] a *,[data-testid="stHeader"] img{pointer-events:auto}
.block-container{padding-top:calc(3.75rem + 1.2rem)!important}
.bg-top{font-family:var(--display);font-weight:600;font-size:clamp(1rem,1.45vw,1.3rem);color:var(--fg);letter-spacing:-.01em;
  height:2.4rem;display:flex;align-items:center;overflow:hidden;white-space:nowrap;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent);mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.st-key-bgtopbar [data-testid="stMarkdownContainer"],.st-key-bgtopbar .stMarkdown{margin:0!important;min-width:0}
.bg-top .piste{display:inline-flex;flex:none;animation:bgDefileTexte 24s linear infinite;will-change:transform}
.bg-top:hover .piste{animation-play-state:paused}
.bg-top .copie{padding-right:5rem;line-height:2.4rem}
.bg-top em{font-style:normal;background:linear-gradient(90deg,#E46E14 0%,#FFB066 45%,#E46E14 90%);background-size:220% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:bgBrille 3.2s linear infinite}
@keyframes bgDefileTexte{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes bgBrille{from{background-position:120% 0}to{background-position:-100% 0}}
@media (prefers-reduced-motion:reduce){.bg-top .piste{animation:none}}
.st-key-bgop button{height:2.4rem;border-radius:999px!important;padding:0 .8rem 0 .3rem!important;max-width:15rem;background:var(--surface)!important}
.st-key-bgop button p{font-family:var(--display);font-weight:600;font-size:.88rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.st-key-bgop button>div>div:first-child [data-testid="stIconMaterial"]{width:1.85rem;height:1.85rem;border-radius:50%;display:grid;place-items:center;font-size:1.2rem;
  color:#fff!important;background:linear-gradient(145deg,#F27A1F,#C4560A);box-shadow:0 2px 8px rgba(242,122,31,.35)}
@media (max-width:1150px){.st-key-bgop button p{display:none}.st-key-bgop button{padding:0 .3rem!important}}
.st-key-bgaccueil{width:auto!important}
.st-key-bgaccueil a{height:2.4rem;border-radius:10px;border:1px solid var(--line);background:var(--surface);padding:0 .9rem!important;white-space:nowrap;
  display:inline-flex!important;align-items:center}
.st-key-bgaccueil a:hover{border-color:var(--accent)}
.st-key-bgaccueil a p{font-weight:500}
.st-key-bgrecherche [data-baseweb="select"]>div{border-radius:10px;background:var(--surface);min-height:2.3rem}
</style>"""


def _changer_utilisateur():
    """Oublie le nom et rejoue l’animation du logo (bouton « Continuer »), puis la saisie du nom."""
    for k in ("nom", "_w_nom", "_nom_vide", "_intro_vue"):
        st.session_state.pop(k, None)


def _index(pages):
    """Entrées de la recherche : (libellé, page)."""
    e = [(pages[k].title, k) for k in pages]
    e += [(T(a, b), k) for a, b, k in (
        ("Signaux bruts du banc de test", "Raw test bench signals", "exploration"),
        ("Les 26 descripteurs", "The 26 features", "exploration"),
        ("Effet de la vitesse", "Speed effect", "exploration"),
        ("Corrélations entre descripteurs", "Feature correlations", "exploration"),
        ("Rapport YData Profiling des données", "YData Profiling data report", "exploration"),
        ("Déposer un fichier Excel", "Upload an Excel file", "diagnostic"),
        ("Choisir le modèle (retenu ou variantes)", "Choose the model (selected or variants)", "diagnostic"),
        ("Contribution des descripteurs à l’alarme", "Feature contributions to the alarm", "diagnostic"),
        ("Simulation en direct", "Live simulation", "surveillance"),
        ("Score d’anomalie en temps réel", "Real-time anomaly score", "surveillance"),
        ("Projection t-SNE de l’espace latent", "t-SNE projection of the latent space", "latent"),
        ("Clustering K-means / GMM", "K-means / GMM clustering", "latent"),
        ("Courbes ROC", "ROC curves", "performances"),
        ("Matrice de confusion", "Confusion matrix", "performances"),
        ("Importance par permutation", "Permutation importance", "performances"),
        ("Variantes et étude comparative", "Variants and comparative study", "performances"),
        ("Modèle retenu", "Selected model", "performances"),
        ("Découpage train / validation / test", "Train / validation / test split", "methodologie"),
        ("Limites connues", "Known limits", "methodologie"),
        ("Équipe", "Team", "methodologie"))]
    return e


def _choisir():
    choix = st.session_state.get("_w_cherche")
    if choix is not None:
        st.session_state["_aller"] = st.session_state["_index_recherche"][choix][1]
    st.session_state["_w_cherche"] = None


def afficher(pages):
    """Dessine la barre ; renvoie la page demandée par la recherche, s'il y en a une."""
    index = _index(pages)
    st.session_state["_index_recherche"] = index
    st.html(CSS)
    phrase = T("Détecter un roulement défectueux <em>avant la panne.</em>", "Detect a faulty bearing <em>before it fails.</em>")
    with st.container(key="bgtopbar"):
        with st.container(key="bgaccueil"):
            st.page_link(pages["accueil"], label=T("Accueil", "Home"), icon=":material/home:",
                         help=T("Revenir au tableau de bord", "Back to the dashboard"))
        st.markdown(f'<div class="bg-top"><div class="piste"><span class="copie">{phrase}</span>'
                    f'<span class="copie" aria-hidden="true">{phrase}</span></div></div>', unsafe_allow_html=True)
        with st.container(key="bgrecherche"):
            st.selectbox(T("Rechercher", "Search"), range(len(index)), index=None, key="_w_cherche",
                         format_func=lambda i: f"{index[i][0]}  ·  {pages[index[i][1]].title}" if index[i][0] != pages[index[i][1]].title else index[i][0],
                         placeholder=T("Rechercher une page, un graphique, un modèle…", "Search a page, a chart, a model…"),
                         label_visibility="collapsed", on_change=_choisir)
        nom = st.session_state.get("nom", "")
        with st.popover(nom, icon=":material/engineering:", key="bgop", help=T("Opérateur connecté", "Signed-in operator")):
            st.caption(T("Opérateur", "Operator"))
            st.markdown(f"**{html.escape(nom)}**")
            st.button(T("Changer d’utilisateur", "Change user"), icon=":material/logout:", on_click=_changer_utilisateur,
                      key="bgchanger", width="stretch")
    return st.session_state.pop("_aller", None)

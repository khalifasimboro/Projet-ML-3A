"""Écran de bienvenue : le logo entre et s'arrête au centre, le visiteur appuie sur « Continuer », puis on demande son nom.

Le nom ne vit que dans st.session_state : il disparaît quand on quitte ou recharge l'application.
"""
import streamlit as st

from ui.i18n import T
from ui.theme import RAIL, data_uri

CSS = """<style>
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stHeader"]{display:none!important}
.stApp,[data-testid="stAppViewContainer"]{background:radial-gradient(60% 55% at 50% 42%,rgba(242,122,31,.16),transparent 70%),__BG__!important}
.block-container{padding-top:0!important}
.st-key-bg_intro{min-height:calc(100vh - 5rem);display:flex!important;flex-direction:column;justify-content:center;align-items:center;gap:1.6rem!important;position:relative}
.bg-splash{position:relative;display:grid;place-items:center;width:100%;padding:2rem 0}
.bg-splash img{width:min(340px,62vw);filter:drop-shadow(0 10px 40px rgba(242,122,31,.35));
  animation:bgEntre 1.8s cubic-bezier(.2,.8,.2,1) both}
.bg-splash .trait{position:absolute;left:0;right:0;top:50%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(242,122,31,.7),transparent);animation:bgTrait 1.8s ease both}
@keyframes bgEntre{0%{transform:translateX(-75vw) scale(.8);opacity:0}100%{transform:none;opacity:1}}
@keyframes bgTrait{0%,30%{opacity:0;transform:scaleX(0)}100%{opacity:.8;transform:scaleX(1)}}
.st-key-bg_continuer{display:flex;justify-content:center;width:auto!important;animation:bgApparait .6s ease 1.6s both}
.st-key-bg_continuer button{background:#F27A1F!important;border-color:#F27A1F!important;padding:.55rem 2.2rem;border-radius:999px;
  box-shadow:0 10px 30px rgba(242,122,31,.35)}
.st-key-bg_continuer button *{color:#fff!important;font-weight:600}
@keyframes bgApparait{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.st-key-bg_accueil_nom{max-width:430px;margin:12vh auto 0;padding:2rem 2rem 1.6rem!important;border-radius:18px!important;
  background:rgba(20,22,28,.82)!important;border:1px solid __LINE__!important;box-shadow:0 30px 80px rgba(0,0,0,.5);
  animation:bgApparait .6s ease both}
.st-key-bg_accueil_nom *{color:__FG__}
.bg-bv{text-align:center}
.bg-bv img{width:150px;margin-bottom:.6rem}
.bg-bv h1{font-family:'Space Grotesk','Segoe UI',sans-serif;font-size:1.55rem!important;font-weight:600;margin:0 0 .25rem;padding:0;color:#fff!important}
.bg-bv p{color:__MUTED__!important;font-size:.88rem;margin:0 0 .4rem}
.st-key-bg_accueil_nom input{background:#0F1116!important;color:#fff!important;border-color:__LINE__!important}
.st-key-bg_accueil_nom [data-testid="stWidgetLabel"] p{color:__MUTED__!important;font-size:.8rem}
.st-key-bg_accueil_nom [data-testid="stFormSubmitButton"] button{background:#F27A1F!important;border-color:#F27A1F!important}
.st-key-bg_accueil_nom [data-testid="stFormSubmitButton"] button *{color:#fff!important}
.st-key-bg_accueil_nom [data-testid="stForm"]{border:none;padding:0}
</style>"""


def ecran():
    """Affiche l'animation puis le formulaire. Enregistre le nom saisi et relance l'application."""
    css = CSS
    for k in ("bg", "line", "fg", "muted"):
        css = css.replace(f"__{k.upper()}__", RAIL[k])
    st.html(css)
    st.sidebar.caption("Ballguard")   # la sidebar existe dès le départ (masquée ici) : elle s'ouvrira dépliée ensuite
    logo = data_uri("logo.png")
    if not st.session_state.get("_intro_vue"):
        with st.container(key="bg_intro"):
            st.html(f'<div class="bg-splash"><div class="trait"></div><img src="{logo}" alt="Ballguard"></div>')
            st.button(T("Continuer", "Continue"), key="bg_continuer", icon=":material/arrow_forward:",
                      on_click=lambda: st.session_state.update(_intro_vue=True))
        return
    with st.container(key="bg_accueil_nom"):
        st.html(f'<div class="bg-bv"><img src="{logo}" alt="Ballguard"><h1>{T("Bienvenue sur Ballguard", "Welcome to Ballguard")}</h1>'
                f'<p>{T("Diagnostic de roulements par autoencodeurs", "Bearing diagnosis with autoencoders")}</p></div>')
        with st.form("bg_form_nom", border=False):
            st.text_input(T("Comment vous appelez-vous ?", "What is your name?"), max_chars=40, key="_w_nom",
                          placeholder=T("Votre prénom", "Your first name"))
            st.form_submit_button(T("Entrer", "Enter"), icon=":material/arrow_forward:", type="primary", width="stretch",
                                  on_click=_valider)
        if st.session_state.get("_nom_vide"):
            st.warning(T("Indiquez votre nom pour continuer.", "Enter your name to continue."))


def _valider():
    """Rappel du bouton : le nom est enregistré avant la relance, l'accueil s'affiche directement."""
    nom = (st.session_state.get("_w_nom") or "").strip()
    st.session_state["_nom_vide"] = not nom
    if nom:
        st.session_state.nom = nom

"""Composants d'interface réutilisés par toutes les pages."""
import math

import plotly.graph_objects as go
import streamlit as st

from .i18n import T, nom_etat, nombre
from .theme import P


def html(code):
    st.markdown(code, unsafe_allow_html=True)


def entete(groupe, titre, accroche=None):
    """Fil d'Ariane (« Ballguard » ramène au tableau de bord), titre et phrase d'accroche de la page."""
    with st.container(key="filariane", horizontal=True, gap=None, vertical_alignment="center"):
        st.page_link(st.session_state["pages"]["accueil"], label="Ballguard", help=T("Retour au tableau de bord", "Back to the dashboard"))
        html(f'<div class="bg-crumb">/ {groupe}</div>')
    html(f'<div class="bg-title">{titre}</div>' + (f'<p class="bg-lead">{accroche}</p>' if accroche else ""))


def titre_carte(titre, sous_titre=None):
    html(f'<p class="bg-h">{titre}</p>' + (f'<p class="bg-sub">{sous_titre}</p>' if sous_titre else ""))


def kpi(col, label, valeur, indice="", unite="", couleur=None, point=None):
    style = f' style="color:{couleur}"' if couleur else ""
    pt = f'<span class="bg-dot {"red" if point == "red" else ""}"></span>' if point else ""
    col.markdown(f'<div class="bg-kpi"><div class="lab">{label}</div><div class="val"{style}>{pt}{valeur}'
                 f'{"<small>" + unite + "</small>" if unite else ""}</div><div class="hint">{indice}</div></div>',
                 unsafe_allow_html=True)


def pastille(texte, couleur=None):
    style = f' style="color:{couleur}"' if couleur else ""
    return f'<span class="bg-pill"{style}>{"<i></i>" if couleur else ""}{texte}</span>'


def pastille_etat(classe):
    return pastille(nom_etat(classe), P()[classe])


def barres(lignes, texte=False):
    """lignes : liste de (libellé, part 0-1, valeur affichée, couleur ou None)."""
    maxi = max((l[1] for l in lignes), default=1) or 1
    corps = "".join(
        f'<div class="b"><span>{l[0]}</span><div class="tr"><i style="width:{l[1] / maxi * 100:.1f}%'
        f'{";background:" + l[3] if len(l) > 3 and l[3] else ""}"></i></div><span>{l[2]}</span></div>' for l in lignes)
    html(f'<div class="bg-bars {"txt" if texte else ""}">{corps}</div>')


def score_txt(s):
    """Score d'anomalie lisible : 3 décimales pour les petits scores, entier au-delà de 100."""
    return nombre(s, 3 if s < 100 else 0)


def jauge(score, seuil):
    """Jauge logarithmique du score d'anomalie, de 0,001 à 100 000."""
    pos = lambda s: max(2, min(98, (math.log10(max(s, 1e-6)) + 3) / 8 * 100))
    html(f'<div class="bg-gauge"><div class="th" style="left:{pos(seuil):.1f}%"><span>{T("seuil", "threshold")} '
         f'{score_txt(seuil)}</span></div><div class="mk" style="left:{pos(score):.1f}%"><span>{score_txt(score)}</span></div></div>')


ICONE_INFO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">'
              '<circle cx="12" cy="12" r="9.5"/><path d="M12 11v6M12 7.5v.5"/></svg>')


def lecture(phrase, definitions, titre=None, avertissement=""):
    """Encadré « Comment lire … » : une phrase en clair, puis la définition de chaque chiffre affiché.

    definitions : liste de (terme, explication).
    """
    defs = "".join(f"<div><b>{t}</b>{d}</div>" for t, d in definitions)
    html(f'<div class="bg-lire"><div class="tt">{ICONE_INFO}{titre or T("Comment lire ces résultats", "How to read these results")}</div>'
         f'<p class="phrase">{phrase}</p>{avertissement}' + (f'<div class="defs">{defs}</div>' if defs else "") + "</div>")


def note(texte):
    html(f'<div class="bg-note">{texte}</div>')


def pied():
    html(f'<div class="bg-foot"><span>Ballguard© · {T("Maintenance prédictive des roulements", "Predictive maintenance for bearings")}</span>'
         f'<span>{T("Une seconde de vibration suffit : Ballguard donne l’alarme, le type de défaut et l’explication.", "One second of vibration is enough: Ballguard gives the alarm, the fault type and the explanation.")}</span></div>')


# ------------------------------------------------------------------ graphiques
def style(fig, hauteur=320, legende=True, marges=None):
    """Applique le thème courant à une figure Plotly."""
    p = P()
    axe = dict(gridcolor=p["line"], zerolinecolor=p["line"], linecolor=p["line"],
               tickfont=dict(color=p["muted"], size=11), title_font=dict(color=p["muted"], size=12))
    fig.update_layout(
        height=hauteur, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=p["fg"], size=12),
        margin=marges or dict(l=50, r=14, t=12, b=44), showlegend=legende,
        legend=dict(orientation="h", y=-0.22, font=dict(color=p["muted"], size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["surface2"], bordercolor=p["line"], font=dict(color=p["fg"])),
        colorway=[p["accent"], p["BALL"], p["SAFE"], p["INNER_RACE"], p["OUTER_RACE"], p["COMBINED"]])
    fig.update_xaxes(**axe)
    fig.update_yaxes(**axe)
    return fig


def tracer(fig, hauteur=320, cle=None, **kw):
    st.plotly_chart(style(fig, hauteur, **kw), use_container_width=True, config={"displayModeBar": False}, key=cle)


def echelle(couleur_haute=None):
    p = P()
    return [[0, p["surface2"]], [1, couleur_haute or p["accent"]]]


def carte_chaleur(z, x, y, hauteur=300, texte=True, cle=None, couleurs=None, marges=None, zmin=None, zmax=None,
                  barre=False):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=couleurs or echelle(), showscale=barre, zmin=zmin, zmax=zmax,
                               texttemplate="%{z}" if texte else None, textfont=dict(size=12), xgap=2, ygap=2,
                               colorbar=dict(thickness=10, tickfont=dict(color=P()["muted"]))))
    fig.update_yaxes(autorange="reversed")
    tracer(fig, hauteur, cle, legende=False, marges=marges or dict(l=120, r=10, t=10, b=40))


def ncle():
    """Clé unique et stable pour une carte : permet de la cibler dans la feuille de style."""
    st.session_state["_n"] = st.session_state.get("_n", 0) + 1
    return f"bgc{st.session_state['_n']}"

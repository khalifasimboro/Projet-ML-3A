"""Exploration des données : base, signaux, descripteurs, effet de la vitesse."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ballguard.config import CLASSES, FS, VITESSES
from ballguard.descripteurs import FEATURES, frequences_spectre
from ui import donnees
from ui.composants import ncle
from ui.composants import carte_chaleur, entete, html, kpi, pied, titre_carte, tracer
from ui.i18n import FORMULES, T, bornes_bande, nom_desc, nom_etat, nombre
from ui.theme import P


def _effectifs(bd):
    p, fig = P(), go.Figure()
    for v, op in zip(VITESSES, (0.45, 0.7, 1)):
        fig.add_bar(name=f"{T('vitesse', 'speed')} {v}", x=[nom_etat(c) for c in CLASSES],
                    y=[int(((bd.classe == c) & (bd.vitesse == v)).sum()) for c in CLASSES],
                    marker=dict(color=[p[c] for c in CLASSES], opacity=op))
    fig.update_layout(barmode="group", yaxis_title=T("fichiers", "files"))
    return fig


def page():
    bd, art, p = donnees.base(), donnees.artefacts(), P()
    entete(T("Données", "Data"), T("Exploration des données", "Data exploration"),
           T("Comprendre la base avant de modéliser : ce qui distingue un roulement sain d’un roulement défectueux, et ce qui peut tromper les modèles.",
             "Understand the data before modelling: what separates a healthy bearing from a faulty one, and what can mislead the models."))
    t1, t2, t3, t4, t5 = st.tabs([T("Vue d’ensemble", "Overview"), T("Signaux", "Signals"),
                                  T("Descripteurs", "Features"), T("Effet de la vitesse", "Speed effect"),
                                  T("Rapport détaillé (YData)", "Detailed report (YData)")],
                                 key="exp_onglets", on_change="rerun")

    with t1:
        c = st.columns(4)
        kpi(c[0], T("Acquisitions", "Acquisitions"), nombre(len(bd), 0), T("1 seconde chacune", "1 second each"))
        kpi(c[1], T("Échantillonnage", "Sampling"), "25,6" if st.session_state.lang == "fr" else "25.6", T("25 600 points par signal", "25,600 points per signal"), " kHz")
        kpi(c[2], T("Descripteurs", "Features"), "26", T("13 par capteur", "13 per sensor"))
        kpi(c[3], T("Découpage", "Split"), "70/15/15", T("Train / validation / test, stratifié", "Train / validation / test, stratified"))
        st.write("")
        g, d = st.columns(2)
        with g.container(border=True, key=ncle()):
            titre_carte(T("Acquisitions par état et vitesse", "Acquisitions per state and speed"))
            tracer(_effectifs(bd), 330)
        with d.container(border=True, key=ncle()):
            proj, var = donnees.acp_train()
            titre_carte(T("Vue d’ensemble en 2D (ACP)", "2D overview (PCA)"), T("Chaque point est un signal du jeu d’entraînement", "Each point is a training signal"))
            ech = proj.sample(min(2500, len(proj)), random_state=0)
            fig = go.Figure()
            for cl in CLASSES:
                s = ech[ech.classe == cl]
                fig.add_scattergl(x=s.c1, y=s.c2, mode="markers", name=nom_etat(cl), marker=dict(color=p[cl], size=5, opacity=0.7),
                                  customdata=s.vitesse, hovertemplate=nom_etat(cl) + " · " + T("vitesse", "speed") + " %{customdata}<extra></extra>")
            fig.update_layout(xaxis_title=f"{T('composante', 'component')} 1 ({var[0]:.0%})", yaxis_title=f"{T('composante', 'component')} 2 ({var[1]:.0%})")
            tracer(fig, 330)

    with t2:
        with st.container(border=True, key=ncle()):
            a, b = st.columns([3, 1])
            etat = a.segmented_control(T("État du roulement", "Bearing state"), CLASSES, format_func=nom_etat, default="INNER_RACE", key="exp_etat") or "INNER_RACE"
            vit = b.segmented_control(T("Vitesse", "Speed"), VITESSES, default=15, key="exp_vit") or 15
            ex = donnees.exemples()
            fichier = ex[(ex.classe == etat) & (ex.vitesse == vit)].fichier.iloc[0]
            sig, _, spectre = donnees.signal_exemple(fichier)
            ref, _, spectre_ref = donnees.signal_exemple(donnees.exemple_sain(vit))
            st.caption(T(f"Signal réel du banc de test : {fichier}, capteur vertbleu. En gris, un roulement sain à la même vitesse.",
                         f"Real test-bench signal: {fichier}, vertbleu sensor. In grey, a healthy bearing at the same speed."))
            g, d = st.columns(2)
            n, t = 2560, np.arange(2560) / FS * 1000
            fig = go.Figure()
            fig.add_scattergl(x=t, y=ref[:n, 0] - ref[:, 0].mean(), name=nom_etat("SAFE"), line=dict(color=p["muted"], width=1))
            if etat != "SAFE":
                fig.add_scattergl(x=t, y=sig[:n, 0] - sig[:, 0].mean(), name=nom_etat(etat), line=dict(color=p[etat], width=1))
            fig.update_layout(xaxis_title=T("temps (ms)", "time (ms)"), yaxis_title=T("accélération", "acceleration"))
            with g:
                titre_carte(T("Signal temporel (100 ms)", "Time signal (100 ms)"))
                tracer(fig, 330)
            f = frequences_spectre() / 1000
            fig = go.Figure()
            fig.add_scatter(x=f, y=10 ** spectre_ref[:256], name=nom_etat("SAFE"), line=dict(color=p["muted"], width=1.2))
            if etat != "SAFE":
                fig.add_scatter(x=f, y=10 ** spectre[:256], name=nom_etat(etat), line=dict(color=p[etat], width=1.3))
            fig.update_layout(xaxis_title=T("fréquence (kHz)", "frequency (kHz)"), yaxis_title=T("puissance", "power"), yaxis_type="log")
            with d:
                titre_carte(T("Densité spectrale de puissance", "Power spectral density"))
                tracer(fig, 330)

    with t3:
        train = bd[bd.split == "train"]
        _glossaire()
        with st.container(border=True, key=ncle()):
            a, b = st.columns([2, 1])
            a.markdown(f'<p class="bg-h">{T("Distribution d’un descripteur par état", "Feature distribution per state")}</p><p class="bg-sub">{T("Jeu d’entraînement uniquement", "Training set only")}</p>', unsafe_allow_html=True)
            desc = b.selectbox(T("Descripteur", "Feature"), FEATURES, index=FEATURES.index("kurtosis_vb"), key="exp_desc", format_func=lambda f: nom_desc(f, True))
            fig = go.Figure()
            for cl in CLASSES:
                fig.add_box(y=train.loc[train.classe == cl, desc], name=nom_etat(cl), marker=dict(color=p[cl], size=3), line=dict(width=1.4), boxpoints="outliers")
            fig.update_layout(yaxis_title=nom_desc(desc), yaxis_type="log" if desc.split("_")[0] in ("kurtosis", "rms", "pkpk", "crete") else "linear")
            tracer(fig, 380, legende=False)
        with st.container(border=True, key=ncle()):
            titre_carte(T("Corrélations entre descripteurs", "Feature correlations"), T("Beaucoup de redondance : l’autoencodeur la compresse", "A lot of redundancy: the autoencoder compresses it"))
            corr = train[FEATURES].corr().round(2)
            noms = [nom_desc(f) for f in FEATURES]
            carte_chaleur(corr.to_numpy(), noms, noms, 620, texte=False, barre=True, zmin=-1, zmax=1,
                          couleurs=[[0, p["BALL"]], [0.5, p["surface2"]], [1, p["accent"]]], marges=dict(l=175, r=10, t=10, b=165))

    with t4:
        train = bd[bd.split == "train"]
        g, d = st.columns(2)
        with g.container(border=True, key=ncle()):
            titre_carte(T("La vitesse, facteur de confusion", "Speed as a confounder"), T("Médiane par état quand la vitesse augmente", "Median per state as speed increases"))
            desc = st.selectbox(T("Descripteur", "Feature"), FEATURES, index=0, key="exp_desc_v", format_func=lambda f: nom_desc(f, True))
            med = train.groupby(["classe", "vitesse"])[desc].median()
            fig = go.Figure()
            for cl in CLASSES:
                fig.add_scatter(x=VITESSES, y=med.loc[cl].reindex(VITESSES), name=nom_etat(cl), mode="lines+markers", line=dict(color=p[cl], width=2), marker=dict(size=7))
            fig.update_layout(xaxis=dict(title=T("vitesse", "speed"), tickvals=VITESSES), yaxis_title=nom_desc(desc))
            tracer(fig, 330)
        with d.container(border=True, key=ncle()):
            eta = art["eta"]
            titre_carte(T("Part de variance expliquée (η²)", "Share of variance explained (η²)"), T("Moyenne sur les 26 descripteurs, avant et après correction", "Average over the 26 features, before and after correction"))
            x = [T("η² vitesse", "η² speed"), T("η² état", "η² state")]
            avant, apres = [np.mean(eta["vitesse_avant"]), np.mean(eta["classe_avant"])], [0.0, np.mean(eta["classe_apres"])]
            fig = go.Figure()
            fig.add_bar(name=T("avant", "before"), x=x, y=avant, marker_color=p["muted"], text=[nombre(v) for v in avant], textposition="outside")
            fig.add_bar(name=T("après correction", "after correction"), x=x, y=apres, marker_color=p["accent"], text=[nombre(v) for v in apres], textposition="outside")
            fig.update_layout(barmode="group", yaxis_range=[0, 1.1])
            tracer(fig, 400)
        with st.container(border=True, key=ncle()):
            titre_carte(T("Descripteur par descripteur", "Feature by feature"), T("Un bon descripteur dépend de l’état, pas de la vitesse", "A good feature depends on the state, not on speed"))
            fig = go.Figure()
            fig.add_bar(name=T("η² vitesse (avant)", "η² speed (before)"), x=[nom_desc(f) for f in eta["features"]], y=eta["vitesse_avant"], marker_color=p["muted"])
            fig.add_bar(name=T("η² état (avant)", "η² state (before)"), x=[nom_desc(f) for f in eta["features"]], y=eta["classe_avant"], marker_color=p["BALL"])
            fig.add_bar(name=T("η² état (après)", "η² state (after)"), x=[nom_desc(f) for f in eta["features"]], y=eta["classe_apres"], marker_color=p["accent"])
            fig.update_layout(barmode="group", yaxis_range=[0, 1.05], xaxis_tickangle=-50)
            tracer(fig, 440, marges=dict(l=40, r=10, t=10, b=170))
    if t5.open:
        with t5:
            _rapport()
    pied()


def _glossaire():
    """Tableau des 13 descripteurs calculés pour chaque capteur : nom, code, formule, sens."""
    sens = {"rms": T("Le niveau moyen de vibration", "The average vibration level"),
            "kurtosis": T("Les chocs : environ 3 pour un roulement sain, bien plus quand un défaut tape", "Shocks: about 3 for a healthy bearing, much more when a fault strikes"),
            "crete": T("La hauteur du plus grand choc par rapport au niveau moyen", "The height of the biggest shock relative to the average level"),
            "skew": T("Si le signal penche d’un côté (0 = symétrique)", "Whether the signal leans to one side (0 = symmetric)"),
            "pkpk": T("L’écart entre le point le plus haut et le plus bas", "The gap between the highest and lowest point")}
    lignes = "".join(f'<tr><td><b>{nom_desc(c + "_vb", True).split(" · ")[0]}</b></td><td><code>{c}</code></td><td>{FORMULES[c]}</td><td>{sens[c]}</td></tr>' for c in sens)
    for b in range(1, 9):
        lo, hi = bornes_bande(b)
        lignes += (f'<tr><td><b>{nom_desc(f"bande{b}_vb").split(" · ")[0]}</b></td><td><code>bande{b}</code></td><td>{FORMULES["bande"]}</td>'
                   f'<td>{T("Énergie des vibrations entre", "Vibration energy between")} {nombre(lo, 2)} {T("et", "and")} {nombre(hi, 1)} kHz</td></tr>')
    with st.container(border=True, key=ncle()):
        titre_carte(T("Les 26 descripteurs", "The 26 features"),
                    T("13 descripteurs par capteur (suffixe VB : accéléromètre vert-bleu, V : accéléromètre vert), calculés sur le signal centré x.",
                      "13 features per sensor (suffix VB: green-blue accelerometer, V: green accelerometer), computed on the centred signal x."))
        html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Nom", "Name")}</th><th>{T("Code", "Code")}</th><th>{T("Formule", "Formula")}</th>'
             f'<th>{T("Ce que ça mesure", "What it measures")}</th></tr>{lignes}</table></div>')
        st.caption(T("Bandes : spectre de Welch (fenêtres de 512 points, une valeur tous les 50 Hz), 256 fréquences de 50 Hz à 12,8 kHz découpées en 8 paquets de 32. "
                     "Le log₁₀ ramène des énergies très différentes sur une même échelle : d’où des valeurs négatives.",
                     "Bands: Welch spectrum (512-point windows, one value every 50 Hz), 256 frequencies from 50 Hz to 12.8 kHz split into 8 groups of 32. "
                     "The log₁₀ brings very different energies onto one scale, hence the negative values."))


RAPPORTS = Path(__file__).resolve().parent.parent / "assets" / "rapports"

# Dans le cadre qui affiche le rapport, une ancre « #overview » serait résolue par rapport à l'adresse de
# l'application, qui se rechargerait alors dans le cadre. Ce script fait défiler jusqu'à la section visée ;
# les onglets, menus et boutons propres au rapport (attribut data-bs-toggle) gardent leur comportement.
ANCRES = """<style>[id]{scroll-margin-top:80px}</style>
<script>
document.addEventListener("click", function (e) {
  var a = e.target.closest("a[href^='#']");
  if (!a || a.hasAttribute("data-bs-toggle") || a.hasAttribute("data-toggle")) return;
  e.preventDefault();
  var id = decodeURIComponent(a.getAttribute("href").slice(1));
  var cible = id && id !== "top" ? (document.getElementById(id) || document.getElementsByName(id)[0]) : null;
  if (cible) cible.scrollIntoView({behavior: "smooth", block: "start"});
  else window.scrollTo({top: 0, behavior: "smooth"});
}, true);
</script>"""


@st.cache_data(show_spinner=False, max_entries=2)
def _rapport_html(chemin):
    contenu = Path(chemin).read_text(encoding="utf-8")
    i = contenu.lower().rfind("</body>")
    return contenu[:i] + ANCRES + contenu[i:] if i >= 0 else contenu + ANCRES


def _rapport():
    """Rapports YData Profiling pré-calculés (scripts/rapports_profiling.py) : une fiche par variable."""
    with st.container(border=True, key=ncle()):
        a, b = st.columns([3, 2], vertical_alignment="center")
        a.markdown(f'<p class="bg-h">{T("Rapport de profilage des données", "Data profiling report")}</p>'
                   f'<p class="bg-sub">{T("Statistiques, distributions, corrélations et alertes pour chaque variable, générés avec YData Profiling.", "Statistics, distributions, correlations and alerts for every variable, generated with YData Profiling.")}</p>',
                   unsafe_allow_html=True)
        choix = b.segmented_control(T("Jeu de données", "Dataset"), ["bruts", "descripteurs"], default="descripteurs", key="exp_rapport",
                                    format_func=lambda c: T("Signaux bruts du banc", "Raw bench signals") if c == "bruts" else T("Base des 26 descripteurs", "26-feature base"),
                                    label_visibility="collapsed") or "descripteurs"
        if choix == "bruts":
            st.caption(T("Les 15 signaux d’exemple publiés (une seconde chacun, 384 000 mesures) : la base brute complète reste privée.",
                         "The 15 published example signals (one second each, 384,000 readings): the full raw base stays private."))
        else:
            st.caption(T("Les 8 431 acquisitions, chacune résumée par 26 descripteurs. La formule de chaque descripteur est rappelée dans sa fiche.",
                         "The 8,431 acquisitions, each summarised by 26 features. Each feature’s formula is given in its card."))
        fichier = RAPPORTS / ("signaux_bruts.html" if choix == "bruts" else "descripteurs.html")
        if fichier.exists():
            st.iframe(_rapport_html(str(fichier)), height=900)
        else:
            st.info(T("Rapport non généré : lancer scripts/rapports_profiling.py.", "Report not generated: run scripts/rapports_profiling.py."))

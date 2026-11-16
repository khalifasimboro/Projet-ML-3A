"""Exploration des données : base, signaux, descripteurs, effet de la vitesse."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ballguard.config import CLASSES, FS, VITESSES
from ballguard.descripteurs import FEATURES, frequences_spectre
from ui import donnees
from ui.composants import ncle
from ui.composants import carte_chaleur, entete, kpi, pied, titre_carte, tracer
from ui.i18n import T, nom_etat, nombre
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
    t1, t2, t3, t4 = st.tabs([T("Vue d’ensemble", "Overview"), T("Signaux", "Signals"),
                              T("Descripteurs", "Features"), T("Effet de la vitesse", "Speed effect")])

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
        with st.container(border=True, key=ncle()):
            a, b = st.columns([2, 1])
            a.markdown(f'<p class="bg-h">{T("Distribution d’un descripteur par état", "Feature distribution per state")}</p><p class="bg-sub">{T("Jeu d’entraînement uniquement", "Training set only")}</p>', unsafe_allow_html=True)
            desc = b.selectbox(T("Descripteur", "Feature"), FEATURES, index=FEATURES.index("kurtosis_vb"), key="exp_desc")
            fig = go.Figure()
            for cl in CLASSES:
                fig.add_box(y=train.loc[train.classe == cl, desc], name=nom_etat(cl), marker=dict(color=p[cl], size=3), line=dict(width=1.4), boxpoints="outliers")
            fig.update_layout(yaxis_title=desc, yaxis_type="log" if desc.split("_")[0] in ("kurtosis", "rms", "pkpk", "crete") else "linear")
            tracer(fig, 380, legende=False)
        with st.container(border=True, key=ncle()):
            titre_carte(T("Corrélations entre descripteurs", "Feature correlations"), T("Beaucoup de redondance : l’autoencodeur la compresse", "A lot of redundancy: the autoencoder compresses it"))
            corr = train[FEATURES].corr().round(2)
            carte_chaleur(corr.to_numpy(), FEATURES, FEATURES, 560, texte=False, barre=True, zmin=-1, zmax=1,
                          couleurs=[[0, p["BALL"]], [0.5, p["surface2"]], [1, p["accent"]]], marges=dict(l=90, r=10, t=10, b=90))

    with t4:
        train = bd[bd.split == "train"]
        g, d = st.columns(2)
        with g.container(border=True, key=ncle()):
            titre_carte(T("La vitesse, facteur de confusion", "Speed as a confounder"), T("Médiane par état quand la vitesse augmente", "Median per state as speed increases"))
            desc = st.selectbox(T("Descripteur", "Feature"), FEATURES, index=0, key="exp_desc_v")
            med = train.groupby(["classe", "vitesse"])[desc].median()
            fig = go.Figure()
            for cl in CLASSES:
                fig.add_scatter(x=VITESSES, y=med.loc[cl].reindex(VITESSES), name=nom_etat(cl), mode="lines+markers", line=dict(color=p[cl], width=2), marker=dict(size=7))
            fig.update_layout(xaxis=dict(title=T("vitesse", "speed"), tickvals=VITESSES), yaxis_title=desc)
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
            fig.add_bar(name=T("η² vitesse (avant)", "η² speed (before)"), x=eta["features"], y=eta["vitesse_avant"], marker_color=p["muted"])
            fig.add_bar(name=T("η² état (avant)", "η² state (before)"), x=eta["features"], y=eta["classe_avant"], marker_color=p["BALL"])
            fig.add_bar(name=T("η² état (après)", "η² state (after)"), x=eta["features"], y=eta["classe_apres"], marker_color=p["accent"])
            fig.update_layout(barmode="group", yaxis_range=[0, 1.05], xaxis_tickangle=-50)
            tracer(fig, 380, marges=dict(l=40, r=10, t=10, b=110))
    pied()

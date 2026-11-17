"""Onglet « Étude comparative V6 » de la page Performances : résultats de l'étude du groupe."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ballguard import config
from ui.composants import carte_chaleur, html, ncle, titre_carte, tracer
from ui.i18n import T, nombre

ALGOS = ["K-means", "GMM", "BIRCH"]
FAMILLES = ["MLP-AE", "Denoising-AE", "VAE", "CNN-AE spectral"]


@st.cache_data(show_spinner=False)
def _csv(nom):
    return pd.read_csv(config.RACINE / "donnees" / "etude_v6" / f"{nom}.csv")


def onglet(p, art):
    st.caption(T("Résultats de l’étude comparative V6 menée par le groupe : 4 autoencodeurs, 3 algorithmes de clustering, audit et tests de robustesse. "
                 "Tableaux repris tels quels (donnees/etude_v6/). Même base et même découpage que l’application.",
                 "Results of the group’s V6 comparative study: 4 autoencoders, 3 clustering algorithms, audit and robustness tests. "
                 "Tables shown as produced (donnees/etude_v6/). Same data and same split as the app."))

    with st.container(border=True, key=ncle()):
        titre_carte(T("Modèles intégrés à l’application", "Models built into the app"),
                    T("Réentraînés par scripts/entrainer.py, évalués sur le jeu de test, sélectionnables dans Diagnostic et Espace latent",
                      "Retrained by scripts/entrainer.py, evaluated on the test set, selectable in Diagnosis and Latent space"))
        lignes = "".join(
            f'<tr class="{"hl" if i == 0 else ""}"><td>{l["variante"]}{" · " + T("retenu", "selected") if i == 0 else ""}</td>'
            f'<td class="n">{nombre(l["ARI"], 3)}</td><td class="n">{nombre(l["NMI"], 3)}</td><td class="n">{nombre(l["silhouette"], 3)}</td>'
            f'<td class="n">{nombre(l["exactitude"] * 100, 1)} %</td></tr>' for i, l in enumerate(art["variantes"]))
        html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Modèle", "Model")}</th><th class="n">ARI</th><th class="n">NMI</th>'
             f'<th class="n">Silhouette</th><th class="n">{T("État correct", "Correct state")}</th></tr>{lignes}</table></div>')

    g, d = st.columns([1.3, 1])
    with g.container(border=True, key=ncle()):
        titre_carte(T("ARI après correction de la vitesse", "ARI after speed correction"),
                    T("4 autoencodeurs × 3 algorithmes de clustering (étude B)", "4 autoencoders × 3 clustering algorithms (study B)"))
        sc = _csv("etude_b_scores_clustering")
        corr = sc[sc.strategie == "correction_vitesse"].pivot(index="famille", columns="algorithme", values="ari").reindex(FAMILLES)[ALGOS]
        carte_chaleur(corr.round(3).to_numpy(), ALGOS, FAMILLES, 290, zmin=0, zmax=1, marges=dict(l=130, r=10, t=10, b=40))
        st.caption(T("Le GMM atteint ARI = 1 avec les quatre autoencodeurs ; BIRCH est instable. Le CNN-AE spectral est parfait avec les trois algorithmes.",
                     "GMM reaches ARI = 1 with all four autoencoders; BIRCH is unstable. The spectral CNN-AE is perfect with all three algorithms."))
    with d.container(border=True, key=ncle()):
        titre_carte(T("Détection : fausses alarmes", "Detection: false alarms"),
                    T("Sur 256 signaux sains du test ; tous détectent 100 % des défauts", "Out of 256 healthy test signals; all detect 100% of faults"))
        da = _csv("etude_a_detection").rename(columns={"famille": "modele", "fp": "fa"})[["modele", "fa"]].assign(ae=True)
        db = _csv("experiences_c_baselines_detection").rename(columns={"methode": "modele", "fausses_alarmes": "fa"})[["modele", "fa"]].assign(ae=False)
        dd = pd.concat([da, db]).sort_values("fa")
        fig = go.Figure(go.Bar(orientation="h", y=dd.modele, x=dd.fa, text=dd.fa, textposition="outside", cliponaxis=False,
                               marker_color=[p["accent"] if a else p["muted"] for a in dd.ae]))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title=T("fausses alarmes (orange : autoencodeurs)", "false alarms (orange: autoencoders)"))
        tracer(fig, 290, legende=False, marges=dict(l=140, r=30, t=10, b=44))

    g, d = st.columns(2)
    with g.container(border=True, key=ncle()):
        titre_carte(T("Effet de la stratégie de vitesse", "Effect of the speed strategy"),
                    T("ARI des autoencodeurs + K-means (étude A ; par vitesse : moyenne des 3 modèles)", "ARI of autoencoders + K-means (study A; per speed: mean of 3 models)"))
        cs = _csv("etude_a_clustering_strategies")
        cs = cs[cs.approche == "autoencodeur"].groupby(["strategie", "famille"]).ari.mean().unstack()
        strat = {"sans_correction": T("Sans correction", "No correction"), "par_vitesse": T("Un modèle par vitesse", "One model per speed"),
                 "correction_vitesse": T("Correction", "Correction")}
        fig = go.Figure()
        for fam, col in zip(FAMILLES, (p["accent"], p["BALL"], p["INNER_RACE"], p["SAFE"])):
            fig.add_bar(name=fam, x=list(strat.values()), y=cs.reindex(list(strat))[fam], marker_color=col)
        fig.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_title="ARI")
        tracer(fig, 320)
    with d.container(border=True, key=ncle()):
        titre_carte(T("Vitesse jamais vue", "Unseen speed"),
                    T("Clustering de référence entraîné sans une vitesse, testé sur elle (étude C)", "Reference clustering trained without one speed, tested on it (study C)"))
        lo = _csv("experiences_c_test_vitesse_inconnue_loso")
        fig = go.Figure()
        for (entree, algo), s in lo.groupby(["entree", "algorithme"]):
            fig.add_bar(name=f"{entree} · {algo}", x=[f"{T('sans', 'without')} {v}" for v in s.vitesse_absente], y=s.ari)
        fig.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_title="ARI")
        tracer(fig, 320)
        st.caption(T("Sur une vitesse non apprise, l’ARI tombe entre 0,20 et 0,78 : la correction de vitesse ne vaut que pour 10, 15 et 20.",
                     "On an unlearned speed, ARI drops to 0.20–0.78: the speed correction only holds for 10, 15 and 20."))

    with st.container(border=True, key=ncle()):
        titre_carte(T("Ce que conclut l’étude", "What the study concludes"))
        points = (
            T("Tous les autoencodeurs détectent 100 % des défauts ; MLP-AE, Denoising-AE et ACP font le moins de fausses alarmes (3).",
              "All autoencoders detect 100% of faults; MLP-AE, Denoising-AE and PCA give the fewest false alarms (3)."),
            T("La correction de la vitesse est le facteur décisif : l’ARI passe d’environ 0,5 à plus de 0,98 pour toutes les familles.",
              "Speed correction is the decisive factor: ARI rises from about 0.5 to above 0.98 for every family."),
            T("Le CNN-AE spectral est le seul parfait avec K-means, GMM et BIRCH, mais il lit 512 valeurs de spectre au lieu de 26 descripteurs.",
              "The spectral CNN-AE is the only one perfect with K-means, GMM and BIRCH, but it reads 512 spectrum values instead of 26 features."),
            T("Un GMM sur les 26 descripteurs corrigés atteint aussi ARI = 1 sans autoencodeur : l’apport de l’autoencodeur n’est pas démontré sur ce banc.",
              "A GMM on the 26 corrected features also reaches ARI = 1 without an autoencoder: the autoencoder’s benefit is not shown on this bench."),
            T("Aucun identifiant de série physique n’existe : l’indépendance entre entraînement et test n’est pas garantie.",
              "There is no physical series identifier: independence between train and test is not guaranteed."))
        st.markdown("\n".join("- " + x for x in points))

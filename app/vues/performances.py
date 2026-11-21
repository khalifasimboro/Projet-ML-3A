"""Performances : autoencodeurs comparés aux modèles de référence, et interprétabilité."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ballguard.config import CLASSES, VITESSES
from ui import donnees
from vues import etude_v6
from ui.composants import ncle
from ui.composants import carte_chaleur, entete, html, lecture, pied, titre_carte, tracer
from ui.i18n import T, nom_desc, nom_etat, nombre
from ui.theme import P

NOMS = {"Isolation Forest": ("Isolation Forest", "Isolation Forest"), "ACP": ("ACP", "PCA"), "CNN-AE spectral": ("CNN-AE spectral", "Spectral CNN-AE"),
        "AE (descripteurs)": ("Autoencodeur (descripteurs)", "Autoencoder (features)"),
        "AE (spectre)": ("Autoencodeur (spectre)", "Autoencoder (spectrum)")}


def page():
    p, art, bd = P(), donnees.artefacts(), donnees.base()
    det, clu = art["detection"], art["clustering"]
    test = (bd.split == "test").to_numpy()
    entete(T("Modèles", "Models"), T("Performances des modèles", "Model performance"),
           T("Les autoencodeurs sont comparés à des modèles de référence simples, sur des signaux jamais vus pendant l’entraînement.",
             "Autoencoders are compared with simple baselines, on signals never seen during training."))
    t1, t2, t3, t4 = st.tabs([T("Détection", "Detection"), "Clustering", T("Interprétabilité", "Interpretability"), T("Étude comparative V6", "V6 comparative study")])

    with t1:
        ae = next(l for l in det["table"] if l["modele"] == "AE (descripteurs)")
        n_sains = sum(ae["confusion"][0])
        lecture(T(f"Question posée : sur <b>{nombre(art['meta']['n_test'], 0)} signaux jamais vus</b> pendant l’entraînement, le modèle sépare-t-il les roulements sains des roulements défectueux ? "
                  f"L’autoencodeur retenu détecte <b>{ae['rappel']:.0%} des défauts</b> et ne se trompe que sur <b>{ae['nb_fausses_alarmes']} roulements sains sur {n_sains}</b>. "
                  "Les modèles de référence, plus simples, servent à vérifier que ce résultat n’est pas trivial.",
                  f"The question: on <b>{nombre(art['meta']['n_test'], 0)} signals never seen</b> during training, does the model separate healthy from faulty bearings? "
                  f"The selected autoencoder detects <b>{ae['rappel']:.0%} of faults</b> and is wrong on only <b>{ae['nb_fausses_alarmes']} healthy bearings out of {n_sains}</b>. "
                  "The simpler baselines check that this result is not trivial."),
                [("AUROC", T("Capacité à classer un défaut au-dessus d’un sain, quel que soit le seuil. 1 = séparation parfaite, 0,5 = hasard.", "Ability to rank a fault above a healthy signal, whatever the threshold. 1 = perfect, 0.5 = chance.")),
                 (T("Rappel", "Recall"), T("Part des roulements défectueux qui déclenchent l’alarme.", "Share of faulty bearings that trigger the alarm.")),
                 (T("Précision", "Precision"), T("Part des alarmes qui correspondent vraiment à un défaut.", "Share of alarms that are real faults.")),
                 (T("Fausses alarmes", "False alarms"), T("Roulements sains signalés à tort : chaque fausse alarme coûte un arrêt inutile.", "Healthy bearings flagged by mistake: each one costs a needless stop."))])
        g, d = st.columns([1.5, 1])
        with g.container(border=True, key=ncle()):
            titre_carte(T("Détection d’anomalie sur le jeu de test", "Anomaly detection on the test set"),
                        T(f"{art['meta']['n_test']} signaux, seuil fixé sur les signaux sains de validation", f"{art['meta']['n_test']} signals, threshold set on healthy validation signals"))
            roles = {"Isolation Forest": T("Référence", "Baseline"), "ACP": T("Référence", "Baseline"),
                     "AE (descripteurs)": T("Modèle retenu", "Selected model"), "AE (spectre)": T("Variante", "Variant"), "CNN-AE spectral": T("Étude V6", "V6 study")}
            lignes = "".join(
                f'<tr class="{"hl" if l["modele"] == "AE (descripteurs)" else ""}"><td>{T(*NOMS[l["modele"]])}</td><td>{roles[l["modele"]]}</td>'
                f'<td class="n">{nombre(l["AUROC"], 3)}</td><td class="n">{nombre(l["rappel"], 3)}</td><td class="n">{nombre(l["precision"], 3)}</td>'
                f'<td class="n">{l["nb_fausses_alarmes"]} ({nombre(l["fausses_alarmes"] * 100, 1)} %)</td></tr>' for l in det["table"])
            html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Modèle", "Model")}</th><th>{T("Rôle", "Role")}</th><th class="n">AUROC</th>'
                 f'<th class="n">{T("Rappel", "Recall")}</th><th class="n">{T("Précision", "Precision")}</th><th class="n">{T("Fausses alarmes", "False alarms")}</th></tr>{lignes}</table></div>')
            st.caption(T("Tous les modèles séparent presque parfaitement sain et défaut sur ces données. L’autoencodeur est retenu pour son erreur interprétable et son espace latent, utile au diagnostic.",
                         "All models separate healthy and faulty almost perfectly on this data. The autoencoder is selected for its interpretable error and its latent space, used for diagnosis."))
        with d.container(border=True, key=ncle()):
            titre_carte(T("Matrice de confusion de l’autoencodeur", "Autoencoder confusion matrix"))
            conf = next(l for l in det["table"] if l["modele"] == "AE (descripteurs)")["confusion"]
            carte_chaleur(conf, [T("prédit : normal", "predicted: normal"), T("prédit : alarme", "predicted: alarm")],
                          [T("réel : sain", "actual: healthy"), T("réel : défaut", "actual: fault")], 270, marges=dict(l=100, r=10, t=10, b=40))
        g, d = st.columns(2)
        with g.container(border=True, key=ncle()):
            titre_carte(T("Courbes ROC", "ROC curves"), T("Zoom sur les faibles taux de fausses alarmes", "Zoom on low false-alarm rates"))
            fig = go.Figure()
            for (nom, (fpr, tpr)), col in zip(det["roc"].items(), (p["muted"], p["BALL"], p["accent"], p["SAFE"], p["INNER_RACE"])):
                fig.add_scatter(x=fpr, y=tpr, name=T(*NOMS[nom]), line=dict(color=col, width=2, shape="hv"))
            fig.update_layout(xaxis=dict(title=T("taux de fausses alarmes", "false-alarm rate"), range=[-0.001, 0.05]),
                              yaxis=dict(title=T("taux de détection", "detection rate"), range=[0.95, 1.003]))
            tracer(fig, 340, marges=dict(l=56, r=10, t=10, b=80))
        with d.container(border=True, key=ncle()):
            titre_carte(T("Scores par état", "Scores per state"), T("Erreur de reconstruction (échelle log) et seuil", "Reconstruction error (log scale) and threshold"))
            fig = go.Figure()
            for cl in CLASSES:
                fig.add_box(y=det["score_ae"][test & (bd.classe == cl).to_numpy()], name=nom_etat(cl), marker=dict(color=p[cl], size=3), line=dict(width=1.4))
            fig.add_hline(y=det["seuils"]["AE (descripteurs)"], line=dict(color=p["fg"], dash="dash", width=1.2))
            fig.update_layout(yaxis=dict(type="log", title="score"))
            tracer(fig, 340, legende=False)
        with st.container(border=True, key=ncle()):
            titre_carte(T("Taux d’alarme par état et par vitesse", "Alarm rate per state and speed"),
                        T("Pour « Sain », c’est le taux de fausses alarmes ; pour les défauts, le taux de détection", "For “Healthy” it is the false-alarm rate; for faults, the detection rate"))
            carte_chaleur((np.array(det["taux_alarme"]) * 100).round(1), [f"{T('vitesse', 'speed')} {v}" for v in VITESSES], [nom_etat(c) for c in CLASSES], 260)

    with t2:
        lecture(T("Question posée : sans connaître les étiquettes, le modèle range-t-il les signaux en groupes qui correspondent aux vrais états du roulement ? "
                  f"Avec la solution retenue, <b>{clu['exactitude']:.1%} des signaux de test</b> sont rangés dans le bon état. "
                  "La difficulté venait de la vitesse : sans correction, les groupes se forment par vitesse plutôt que par défaut.",
                  "The question: without knowing the labels, does the model group signals according to the real bearing states? "
                  f"With the selected solution, <b>{clu['exactitude']:.1%} of test signals</b> land in the right state. "
                  "The difficulty was speed: without correction, groups form by speed rather than by fault."),
                [("ARI", T("Accord entre les groupes trouvés et les vrais états. 1 = identiques, 0 = hasard.", "Agreement between the groups found and the real states. 1 = identical, 0 = chance.")),
                 ("NMI", T("Information commune aux groupes et aux états, de 0 à 1.", "Information shared by groups and states, from 0 to 1.")),
                 ("Silhouette", T("Netteté des groupes : proche de 1 si chaque groupe est compact et bien séparé des autres.", "Group sharpness: close to 1 when each group is compact and well separated.")),
                 (T("ARI vs vitesse", "ARI vs speed"), T("Accord entre les groupes et la vitesse. Il doit rester proche de 0 : les groupes ne doivent pas refléter la vitesse.", "Agreement between groups and speed. It should stay close to 0: groups must not reflect speed."))])
        pv = clu["par_vitesse"]
        moy = lambda m, k: float(np.mean([l[k] for l in pv if l["modele"] == m]))
        strategies = [(T("Sans correction", "No correction"), clu["sans"]["baseline"], clu["sans"]["autoencodeur"]),
                      (T("Un modèle par vitesse", "One model per speed"), {k: moy("baseline", k) for k in ("ARI", "NMI", "silhouette")},
                       {k: moy("autoencodeur", k) for k in ("ARI", "NMI", "silhouette")}),
                      (T("Log + standardisation par vitesse", "Log + per-speed scaling"), clu["corrige"]["baseline"], clu["corrige"]["autoencodeur"])]
        with st.container(border=True, key=ncle()):
            titre_carte(T("Trois stratégies de clustering", "Three clustering strategies"), T("k-means à 5 groupes, évalué sur le jeu de test", "k-means with 5 groups, evaluated on the test set"))
            x = [s[0] for s in strategies]
            fig = go.Figure()
            for nom, idx, cle, col, op in (("k-means · ARI", 1, "ARI", p["muted"], 1), (T("Autoencodeur · ARI", "Autoencoder · ARI"), 2, "ARI", p["accent"], 1),
                                            ("k-means · silhouette", 1, "silhouette", p["muted"], 0.45), (T("Autoencodeur · silhouette", "Autoencoder · silhouette"), 2, "silhouette", p["accent"], 0.5)):
                y = [s[idx][cle] for s in strategies]
                fig.add_bar(name=nom, x=x, y=y, marker=dict(color=col, opacity=op), text=[nombre(v) for v in y], textposition="outside")
            fig.update_layout(barmode="group", yaxis_range=[0, 1.15])
            tracer(fig, 400, marges=dict(l=40, r=10, t=10, b=80))
        with st.container(border=True, key=ncle()):
            f = lambda r, k: nombre(r[k], 3) if k in r else "—"
            lignes = "".join(
                f'<tr class="{"hl" if i == 2 and j == 1 else ""}"><td>{s[0]}</td><td>{"k-means" if j == 0 else T("Autoencodeur + k-means", "Autoencoder + k-means")}</td>'
                f'<td class="n">{f(r, "ARI")}</td><td class="n">{f(r, "NMI")}</td><td class="n">{f(r, "silhouette")}</td><td class="n">{f(r, "ARI_vitesse")}</td></tr>'
                for i, s in enumerate(strategies) for j, r in enumerate(s[1:]))
            html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Stratégie", "Strategy")}</th><th>{T("Modèle", "Model")}</th><th class="n">ARI</th><th class="n">NMI</th>'
                 f'<th class="n">Silhouette</th><th class="n">ARI {T("vs vitesse", "vs speed")}</th></tr>{lignes}</table></div>')
            st.caption(T(f"Avec la solution retenue, l’état est correctement identifié pour {clu['exactitude']:.1%} des signaux de test. L’essentiel du gain vient du prétraitement ; l’autoencodeur apporte des groupes plus nets (silhouette).",
                         f"With the selected solution, the state is correctly identified for {clu['exactitude']:.1%} of test signals. Most of the gain comes from preprocessing; the autoencoder brings sharper groups (silhouette)."))

    with t3:
        lecture(T("Question posée : sur quels descripteurs le modèle s’appuie-t-il ? On mélange au hasard un descripteur à la fois et on mesure combien la performance chute. "
                  "Plus la chute est forte, plus le modèle dépend de ce descripteur. La carte du bas montre, pour chaque type de défaut, les descripteurs que l’autoencodeur reconstruit le plus mal.",
                  "The question: which features does the model rely on? One feature at a time is shuffled at random and the drop in performance is measured. "
                  "The bigger the drop, the more the model depends on that feature. The map below shows, for each fault type, the features the autoencoder reconstructs worst."),
                [], titre=T("Comment lire cette page", "How to read this page"))
        g, d = st.columns(2)
        for col, cle, titre, sous, axe in (
                (g, "clustering", T("Importance par permutation : diagnostic", "Permutation importance: diagnosis"),
                 T("Baisse de l’ARI quand un descripteur est mélangé au hasard", "ARI drop when a feature is randomly shuffled"), T("baisse de l’ARI", "ARI drop")),
                (d, "detection", T("Importance par permutation : détection", "Permutation importance: detection"),
                 T("Baisse du score moyen des défauts (log10) quand un descripteur est mélangé", "Drop in mean fault score (log10) when a feature is shuffled"), T("baisse du score (log10)", "score drop (log10)"))):
            with col.container(border=True, key=ncle()):
                titre_carte(titre, sous)
                top = sorted(art["permutation"][cle].items(), key=lambda kv: -kv[1])[:12]
                fig = go.Figure(go.Bar(orientation="h", y=[nom_desc(k) for k, _ in top], x=[v for _, v in top], marker_color=p["accent"]))
                fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
                fig.update_xaxes(title=axe)
                tracer(fig, 400, legende=False, marges=dict(l=175, r=16, t=10, b=44))
        with st.container(border=True, key=ncle()):
            feats = donnees.modele()["features"]
            titre_carte(T("Erreur de reconstruction par descripteur et par état", "Reconstruction error per feature and state"),
                        T("Quels descripteurs trahissent chaque défaut (log10 de l’erreur moyenne, jeu de test)", "Which features give each fault away (log10 of mean error, test set)"))
            carte_chaleur(np.log10(np.array(det["erreur_par_classe"])).round(2), [nom_desc(f) for f in feats], [nom_etat(c) for c in CLASSES], 400, texte=False, barre=True,
                          couleurs=[[0, p["surface2"]], [0.5, p["INNER_RACE"]], [1, p["accent"]]], marges=dict(l=120, r=8, t=10, b=165))
    with t4:
        etude_v6.onglet(p, art)
    pied()

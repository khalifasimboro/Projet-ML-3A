"""Méthodologie et à propos."""
import streamlit as st

from ui import donnees
from ui.composants import ncle
from ui.composants import entete, html, pastille, pied, titre_carte
from ui.i18n import T, nombre

DEPOT = "https://github.com/khalifasimboro/Projet-ML-3A/tree/Rydouane"


def page():
    art = donnees.artefacts()
    entete(T("Projet", "Project"), T("Méthodologie et à propos", "Methodology & about"),
           T("Comment Ballguard a été construit, sur quelles données, et ce qu’il faut garder en tête en lisant ses résultats.",
             "How Ballguard was built, on which data, and what to keep in mind when reading its results."))
    etapes = [
        (T("Découpage avant tout prétraitement", "Split before any preprocessing"),
         T("70 % entraînement, 15 % validation, 15 % test, stratifié par état et vitesse. Toute normalisation est apprise sur l’entraînement seul, le seuil sur la validation.",
           "70% train, 15% validation, 15% test, stratified by state and speed. Every scaler is fitted on train only, the threshold on validation.")),
        (T("Du signal aux descripteurs", "From signal to features"),
         T("Chaque seconde de vibration est résumée par 13 descripteurs par capteur : niveau vibratoire (RMS), impulsivité (kurtosis), facteur de crête, asymétrie, amplitude crête-à-crête et l’énergie dans 8 bandes de 1,6 kHz (de 0,05 à 12,8 kHz).",
           "Each second of vibration is summarised by 13 features per sensor: vibration level (RMS), impulsiveness (kurtosis), crest factor, asymmetry, peak-to-peak amplitude and the energy in 8 bands of 1.6 kHz (0.05 to 12.8 kHz).")),
        (T("Détection : apprendre le sain", "Detection: learn what healthy looks like"),
         T("Un autoencodeur (26 → 16 → 6 → 16 → 26) entraîné sur les seuls roulements sains. Un signal mal reconstruit est une anomalie.",
           "An autoencoder (26 → 16 → 6 → 16 → 26) trained on healthy bearings only. A poorly reconstructed signal is an anomaly.")),
        (T("Corriger l’effet de la vitesse", "Correct the speed effect"),
         T("Passage au logarithme puis standardisation par vitesse, pour que les groupes reflètent le défaut et non la vitesse.",
           "Log transform then per-speed scaling, so groups reflect the fault rather than the speed.")),
        (T("Diagnostic : regrouper sans étiquettes", "Diagnosis: group without labels"),
         T("k-means à 5 groupes sur l’espace latent d’un second autoencodeur. Les étiquettes servent seulement à nommer les groupes.",
           "k-means with 5 groups on the latent space of a second autoencoder. Labels are only used to name the groups.")),
        (T("Comparer et expliquer", "Compare and explain"),
         T("Modèles de référence (Isolation Forest, ACP, k-means direct), décomposition de l’erreur par descripteur et importance par permutation.",
           "Baselines (Isolation Forest, PCA, plain k-means), per-feature error breakdown and permutation importance.")),
    ]
    g, d = st.columns([1.6, 1])
    with g.container(border=True, key=ncle()):
        titre_carte(T("La démarche, étape par étape", "The approach, step by step"))
        html('<div class="bg-steps">' + "".join(f'<div class="s"><div class="n">{i + 1}</div><div><b>{t}</b><p>{x}</p></div></div>' for i, (t, x) in enumerate(etapes)) + "</div>")
    with d:
        with st.container(border=True, key=ncle()):
            titre_carte(T("Données", "Data"))
            m = art["meta"]
            lignes = [(T("Source", "Source"), f"<b>{T('Banc de test', 'Test bench')}</b>"), (T("Signaux exploités", "Signals used"), nombre(m["n"], 0)),
                      (T("États", "States"), T("1 sain, 4 défauts", "1 healthy, 4 faults")), (T("Vitesses", "Speeds"), "10 · 15 · 20"),
                      (T("Capteurs", "Sensors"), T("2 accéléromètres", "2 accelerometers")), (T("Échantillonnage", "Sampling"), "25,6 kHz · 1 s"),
                      ("Train / validation / test", f"{m['n_train']} / {m['n_val']} / {m['n_test']}")]
            html('<table class="bg-table">' + "".join(f'<tr><td>{a}</td><td class="n">{b}</td></tr>' for a, b in lignes) + "</table>")
        with st.container(border=True, key=ncle()):
            titre_carte(T("Limites connues", "Known limits"))
            st.markdown("\n".join("- " + x for x in (
                T("Les scores sont optimistes : des acquisitions consécutives, presque identiques, se retrouvent dans l’entraînement et le test.",
                  "Scores are optimistic: consecutive, near-identical acquisitions end up in both train and test."),
                T("La vitesse doit être connue et faire partie des trois vitesses apprises.", "Speed must be known and be one of the three learned speeds."),
                T("Un seul banc de test : la généralisation à d’autres machines n’est pas mesurée.", "A single test bench: generalisation to other machines is not measured."))))
        with st.container(border=True, key=ncle()):
            titre_carte(T("Technologies", "Stack"))
            html("".join(pastille(x) for x in ("Python 3.11", f"scikit-learn {art['meta']['version_sklearn']}", "SciPy", "pandas", "Plotly", "Streamlit")))
            st.link_button(T("Voir le dépôt GitHub", "Open the GitHub repository"), DEPOT, icon=":material/code:")
    with st.container(border=True, key=ncle()):
        titre_carte(T("Équipe", "Team"), T("Projet ML, sujet 6", "ML project, topic 6"))
        html('<div class="bg-people">' + "".join(f'<div>{n}<span>{T("Élève ingénieur", "Engineering student")}</span></div>'
                                                 for n in ("KINDO Harouna", "HODONOU Firmin", "DOMBOUÉ Tidiane", "SIMBORO Rydouane")) + "</div>")
    pied()

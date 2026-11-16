"""Accueil : l'essentiel en un écran."""
import streamlit as st

from ui import donnees
from ui.composants import ncle
from ui.composants import html, kpi, pied, titre_carte
from ui.i18n import T, nombre


def page():
    art, pages = donnees.artefacts(), st.session_state["pages"]
    det = next(l for l in art["detection"]["table"] if l["modele"] == "AE (descripteurs)")
    n_sains = sum(det["confusion"][0])

    gauche, droite = st.columns([3, 2], vertical_alignment="bottom")
    with gauche:
        html(f'<div class="bg-intro"><h2>{T("Détecter un roulement défectueux <em>avant la panne</em>.", "Detect a faulty bearing <em>before it fails</em>.")}</h2>'
             f'<p class="bg-lead">{T("Une seconde de vibration suffit : Ballguard donne l’alarme, le type de défaut et l’explication.", "One second of vibration is enough: Ballguard gives the alarm, the fault type and the explanation.")}</p></div>')
    with droite:
        a, b = st.columns(2)
        a.page_link(pages["diagnostic"], label=T("Analyser un signal", "Analyse a signal"), icon=":material/troubleshoot:")
        b.page_link(pages["surveillance"], label=T("Surveillance", "Monitoring"), icon=":material/monitor_heart:")

    c = st.columns(4)
    kpi(c[0], T("Signaux analysés", "Signals analysed"), nombre(art["meta"]["n"], 0), T("5 états · 3 vitesses", "5 states · 3 speeds"))
    kpi(c[1], T("Défauts détectés", "Faults detected"), nombre(det["rappel"] * 100, 0), T("Jeu de test", "Test set"), " %")
    kpi(c[2], T("Fausses alarmes", "False alarms"), nombre(det["fausses_alarmes"] * 100, 1),
        T(f"{det['nb_fausses_alarmes']} signaux sains sur {n_sains}", f"{det['nb_fausses_alarmes']} healthy signals out of {n_sains}"), " %")
    kpi(c[3], T("Accord clusters / états", "Cluster / state agreement"), nombre(art["clustering"]["corrige"]["autoencodeur"]["ARI"], 3), "ARI")

    st.write("")
    with st.container(border=True, key=ncle()):
        titre_carte(T("Du signal brut au diagnostic", "From raw signal to diagnosis"))
        etapes = [(T("Signal brut", "Raw signal"), T("Fichier Excel, 1 seconde", "Excel file, 1 second")),
                  (T("26 descripteurs", "26 features"), T("Temps et fréquence", "Time and frequency")),
                  (T("Autoencodeur", "Autoencoder"), T("Appris sur le sain", "Trained on healthy")),
                  (T("Détection", "Detection"), T("Sain ou défaut", "Healthy or fault")),
                  (T("Diagnostic", "Diagnosis"), T("Type de défaut", "Fault type"))]
        html('<div class="bg-pipe">' + '<div class="ar">→</div>'.join(
            f'<div class="st"><b>{t}</b><span>{s}</span></div>' for t, s in etapes) + "</div>")

    modules = [("exploration", "▥", T("Signaux, descripteurs, effet de la vitesse, corrélations.", "Signals, features, speed effect, correlations.")),
               ("diagnostic", "⌖", T("Déposez un fichier Excel : alarme, type de défaut et explication.", "Drop an Excel file: alarm, fault type and explanation.")),
               ("surveillance", "∿", T("Rejouez une série d’acquisitions et suivez le score en direct.", "Replay an acquisition series and follow the score live.")),
               ("latent", "◎", T("Visualisez comment l’autoencodeur organise les signaux.", "See how the autoencoder organises the signals.")),
               ("performances", "▲", T("Autoencodeurs comparés aux modèles de référence.", "Autoencoders compared with baselines.")),
               ("methodologie", "❏", T("Démarche, données, limites connues, équipe.", "Approach, data, known limits, team."))]
    for ligne in (modules[:3], modules[3:]):
        for col, (cle, icone, texte) in zip(st.columns(3), ligne):
            with col.container(border=True, key=ncle()):
                html(f'<div class="bg-mod"><div class="ic">{icone}</div><p class="bg-h">{pages[cle].title}</p><p>{texte}</p></div>')
                st.page_link(pages[cle], label=T("Ouvrir", "Open"), icon=":material/arrow_forward:")
    pied()

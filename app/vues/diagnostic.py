"""Diagnostic d'un signal : du fichier Excel brut à la décision expliquée."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ballguard import config
from ballguard.config import CLASSES, FS, VITESSES
from ballguard.descripteurs import frequences_spectre
from ballguard.diagnostic import VitesseInconnue, diagnostiquer_fichier
from ballguard.lecture import SignalInvalide
from ui import donnees
from ui.composants import ncle
from ui.composants import barres, entete, html, jauge, note, pastille, pastille_etat, pied, score_txt, titre_carte, tracer
from ui.i18n import T, nom_desc, nom_etat, nombre
from ui.theme import P


@st.cache_data(show_spinner=False, max_entries=30)
def _analyser(nom, contenu, vitesse, variante):
    """Lecture + 26 descripteurs + spectre + modèle. Mis en cache par (fichier, vitesse, modèle)."""
    return diagnostiquer_fichier(donnees.modele(), io.BytesIO(contenu), vitesse, variante)


def noms_variantes():
    return {"km": T("MLP-AE + K-means (retenu)", "MLP-AE + K-means (selected)"), "gmm": "MLP-AE + GMM",
            "cnn": T("CNN-AE spectral + K-means", "Spectral CNN-AE + K-means")}


def _rapport(nom, vitesse, r):
    lignes = [f"Ballguard - {T('rapport de diagnostic', 'diagnosis report')}", "=" * 46,
              f"{T('Fichier', 'File')} : {nom}", f"{T('Vitesse', 'Speed')} : {vitesse}",
              f"{T('Verdict', 'Verdict')} : {T('DEFAUT DETECTE', 'FAULT DETECTED') if r['alarme'] else T('ROULEMENT SAIN', 'HEALTHY BEARING')}",
              f"{T('Etat estime', 'Estimated state')} : {nom_etat(r['etat_estime'])}",
              f"{T('Modele', 'Model')} : {noms_variantes()[r['variante']]}",
              f"{T('Score anomalie', 'Anomaly score')} : {r['score']:.4f} ({T('seuil', 'threshold')} {r['seuil']:.4f})", "",
              T("Modeles compares :", "Models compared:")]
    lignes += [f"  - {c['modele']}: {'ALARME' if c['alarme'] else 'normal'} (score/seuil x{c['score'] / c['seuil']:.2f})" for c in r["comparaison"]]
    lignes += ["", T("Descripteurs qui portent l'erreur de reconstruction :", "Features carrying the reconstruction error:")]
    lignes += [f"  - {nom_desc(f, True)}: {v:.1%}" for f, v in list(r["contributions"].items())[:8]]
    lignes += ["", T("Descripteurs (valeur | plage saine a cette vitesse) :", "Features (value | healthy range at this speed):")]
    lignes += [f"  {nom_desc(f):<30} {v:>10.4f} | {r['plage_saine'][f][0]:.4f} - {r['plage_saine'][f][1]:.4f}" for f, v in r["descripteurs"].items()]
    return "\n".join(lignes)


def page():
    p = P()
    entete(T("Analyse", "Analysis"), T("Diagnostic d’un signal", "Signal diagnosis"),
           T("Déposez un signal brut : l’application calcule les 26 descripteurs, interroge le modèle et explique sa décision.",
             "Drop a raw signal: the app computes the 26 features, queries the model and explains its decision."))
    ex = donnees.exemples()
    gauche, droite = st.columns([1, 2.6], gap="medium")

    with gauche.container(border=True, key=ncle()):
        titre_carte(T("Signal à analyser", "Signal to analyse"))
        televerses = st.file_uploader(T("Fichiers Excel du banc de test (.xlsx, 25 600 points)", "Test-bench Excel files (.xlsx, 25,600 points)"),
                                      type=["xlsx"], accept_multiple_files=True, key="diag_up")
        if televerses:
            vitesse = st.segmented_control(T("Vitesse de rotation", "Rotation speed"), VITESSES, default=15, key="diag_vit") or 15
            sources = [(f.name, f.getvalue(), vitesse) for f in televerses]
        else:
            choix = st.selectbox(T("Ou choisissez un signal d’exemple", "Or pick a sample signal"), ex.index, index=int(ex.index[(ex.classe == "INNER_RACE") & (ex.vitesse == 15)][0]),
                                 format_func=lambda i: f"{nom_etat(ex.classe[i])} · {T('vitesse', 'speed')} {ex.vitesse[i]}", key="diag_ex")
            l = ex.loc[choix]
            st.caption(f"{l.fichier} · {T('vitesse', 'speed')} {l.vitesse}")
            sources = [(l.fichier, (config.DOSSIER_EXEMPLES / l.fichier).read_bytes(), int(l.vitesse))]
        noms = noms_variantes()
        variante = st.radio(T("Modèle de diagnostic", "Diagnosis model"), list(noms), format_func=noms.get, key="diag_var",
                            help=T("Le modèle retenu reste le MLP-AE + K-means. Les deux autres viennent de l’étude comparative V6 du groupe.",
                                   "The selected model remains MLP-AE + K-means. The other two come from the group’s V6 comparative study."))
        note(T("Étapes : lecture du fichier → 26 descripteurs et spectre → autoencodeur de détection → encodeur + clustering.",
               "Steps: file read → 26 features and spectrum → detection autoencoder → encoder + clustering."))

    resultats, erreurs = {}, {}
    with droite:
        with st.spinner(T("Analyse du signal…", "Analysing the signal…")):
            for nom, contenu, v in sources:
                try:
                    resultats[nom] = _analyser(nom, contenu, v, variante)
                except (SignalInvalide, VitesseInconnue) as e:
                    erreurs[nom] = str(e)
        for nom, msg in erreurs.items():
            st.error(f"**{nom}** : {T('fichier refusé', 'file rejected')}. {msg}", icon=":material/error:")
        if not resultats:
            st.info(T("Aucun signal exploitable pour l’instant.", "No usable signal yet."))
            return
        vitesse = sources[0][2]

        if len(resultats) > 1:
            with st.container(border=True, key=ncle()):
                titre_carte(T("Résultats du lot", "Batch results"), T(f"{len(resultats)} signaux analysés", f"{len(resultats)} signals analysed"))
                table = pd.DataFrame([{T("fichier", "file"): n, T("verdict", "verdict"): T("Défaut", "Fault") if r["alarme"] else T("Sain", "Healthy"),
                                       T("état estimé", "estimated state"): nom_etat(r["etat_estime"]), "score": round(r["score"], 4),
                                       T("score / seuil", "score / threshold"): round(r["score"] / r["seuil"], 2)} for n, (r, _, _) in resultats.items()])
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.download_button(T("Télécharger le tableau (CSV)", "Download the table (CSV)"), table.to_csv(index=False).encode("utf-8-sig"),
                                   "ballguard_resultats.csv", "text/csv", icon=":material/download:")
            nom = st.selectbox(T("Signal à détailler", "Signal to detail"), list(resultats), key="diag_sel")
        else:
            nom = next(iter(resultats))
        r, sig, spectre = resultats[nom]
        mauvais = r["alarme"]

        html(f'<div class="bg-verdict {"bad" if mauvais else "good"}"><div class="ic">{"!" if mauvais else "✓"}</div>'
             f'<div><div class="f">{nom}</div><div class="big">{T("Défaut détecté", "Fault detected") if mauvais else T("Roulement sain", "Healthy bearing")}</div>'
             f'<div style="margin-top:.4rem">{pastille_etat(r["etat_estime"])}{pastille(T("vitesse", "speed") + f" {vitesse}")}{pastille(noms[variante])}</div></div>'
             f'<div class="meta"><div>Score<b>{score_txt(r["score"])}</b></div><div>{T("Seuil", "Threshold")}<b>{nombre(r["seuil"], 3)}</b></div>'
             f'<div>{T("Rapport", "Ratio")}<b>×{nombre(r["score"] / r["seuil"], 1 if r["score"] / r["seuil"] < 100 else 0)}</b></div></div></div>')

        a, b = st.columns(2)
        with a.container(border=True, key=ncle()):
            titre_carte(T("Score d’anomalie", "Anomaly score"), (T("Erreur de reconstruction du CNN sur le spectre, échelle logarithmique", "CNN reconstruction error on the spectrum, log scale") if variante == "cnn"
                        else T("Erreur de reconstruction de l’autoencodeur, échelle logarithmique", "Autoencoder reconstruction error, log scale")))
            jauge(r["score"], r["seuil"])
            lignes = "".join(
                f'<tr class="{"hl" if c["modele"] == ("CNN-AE spectral" if variante == "cnn" else "Autoencodeur") else ""}"><td>{T("Autoencodeur", "Autoencoder") if c["modele"] == "Autoencodeur" else c["modele"]}</td>'
                f'<td>{pastille(T("Alarme", "Alarm"), p["alarm"]) if c["alarme"] else pastille("Normal", p["ok"])}</td>'
                f'<td class="n">×{nombre(c["score"] / c["seuil"], 2 if c["score"] / c["seuil"] < 100 else 0)}</td></tr>' for c in r["comparaison"])
            html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Modèle", "Model")}</th><th>Verdict</th><th class="n">{T("× seuil", "× threshold")}</th></tr>{lignes}</table></div>')
        with b.container(border=True, key=ncle()):
            titre_carte(T("Pourquoi ce verdict ?", "Why this verdict?"), T("Part de l’erreur de reconstruction portée par chaque descripteur (autoencodeur MLP)", "Share of the reconstruction error carried by each feature (MLP autoencoder)"))
            if variante == "cnn":
                st.caption(T("Le CNN lit le spectre entier : son erreur ne se décompose pas par descripteur. Voici l’explication du MLP-AE pour le même signal.",
                             "The CNN reads the whole spectrum: its error does not split by feature. Here is the MLP-AE explanation for the same signal."))
            hors = [f for f, v in r["descripteurs"].items() if not r["plage_saine"][f][0] <= v <= r["plage_saine"][f][1]]
            barres([(nom_desc(f), v, f"{v:.0%}") for f, v in list(r["contributions"].items())[:7]])
            st.write("")
            st.caption(T(f"{len(hors)} descripteur(s) sur 26 hors de la plage saine à cette vitesse.", f"{len(hors)} of 26 feature(s) outside the healthy range at this speed."))

        ref_sig, _, _ = donnees.signal_exemple(donnees.exemple_sain(vitesse))
        ref_spec = donnees.modele()["reference"][int(vitesse)]["spectre"]
        coul = p["alarm"] if mauvais else p["accent"]
        a, b = st.columns(2)
        with a.container(border=True, key=ncle()):
            titre_carte(T("Signal temporel", "Time signal"), T("100 ms, comparé à un roulement sain à la même vitesse", "100 ms, compared with a healthy bearing at the same speed"))
            n, t = 2560, np.arange(2560) / FS * 1000
            fig = go.Figure()
            fig.add_scattergl(x=t, y=sig[:n, 0] - sig[:, 0].mean(), name=nom, line=dict(color=coul, width=1))
            fig.add_scattergl(x=t, y=ref_sig[:n, 0] - ref_sig[:, 0].mean(), name=T("référence saine", "healthy reference"), line=dict(color=p["muted"], width=1))
            fig.update_layout(xaxis_title="ms")
            tracer(fig, 280)
        with b.container(border=True, key=ncle()):
            titre_carte(T("Spectre", "Spectrum"), T("Densité spectrale (Welch) et moyenne des roulements sains", "Spectral density (Welch) and healthy-bearing average"))
            f = frequences_spectre() / 1000
            fig = go.Figure()
            fig.add_scatter(x=f, y=10 ** spectre[:256], name=nom, line=dict(color=coul, width=1.3))
            fig.add_scatter(x=f, y=10 ** ref_spec[:256], name=T("référence saine", "healthy reference"), line=dict(color=p["muted"], width=1.2))
            fig.update_layout(xaxis_title="kHz", yaxis_type="log")
            tracer(fig, 280)

        a, b = st.columns([1.2, 1])
        with a.container(border=True, key=ncle()):
            titre_carte(T("Type de défaut estimé", "Estimated fault type"), (T("Probabilité d’appartenance à chaque groupe (GMM)", "Membership probability for each group (GMM)") if variante == "gmm"
                        else T("Proximité du signal à chaque groupe de l’espace latent", "Closeness of the signal to each latent-space group")))
            barres([(nom_etat(c), r["proximite"][c], f"{r['proximite'][c]:.0%}", p[c]) for c in CLASSES], texte=True)
        with b.container(border=True, key=ncle()):
            titre_carte(T("Exporter", "Export"), T("Rapport texte du diagnostic affiché", "Text report of the diagnosis shown"))
            st.download_button(T("Télécharger le rapport", "Download the report"), _rapport(nom, vitesse, r).encode("utf-8"),
                               f"ballguard_{nom.rsplit('.', 1)[0]}.txt", "text/plain", icon=":material/description:")
        with st.expander(T("Les 26 descripteurs calculés", "The 26 computed features")):
            st.dataframe(pd.DataFrame([{T("descripteur", "feature"): nom_desc(f, True), T("code", "code"): f, T("valeur", "value"): v, T("plage saine (min)", "healthy range (min)"): r["plage_saine"][f][0],
                                        T("plage saine (max)", "healthy range (max)"): r["plage_saine"][f][1],
                                        T("dans la plage", "in range"): r["plage_saine"][f][0] <= v <= r["plage_saine"][f][1]} for f, v in r["descripteurs"].items()]),
                         hide_index=True, use_container_width=True)
    pied()

"""Diagnostic d'un signal : du fichier Excel brut à la décision expliquée."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ballguard import config
from ballguard.config import CLASSES, FS, VITESSES
from ballguard import vitesse as reco_vitesse
from ballguard.descripteurs import extraire_depuis_signal, frequences_spectre
from ballguard.diagnostic import VitesseInconnue, diagnostiquer_fichier
from ballguard.lecture import SignalInvalide, lire_signal
from ui import donnees
from ui.composants import ncle
from ui.composants import barres, entete, html, jauge, lecture, note, pastille, pastille_etat, pied, score_txt, titre_carte, tracer
from ui.i18n import T, nom_desc, nom_etat, nombre
from ui.theme import P


@st.cache_data(show_spinner=False, max_entries=30)
def _analyser(nom, contenu, vitesse, variante):
    """Lecture + 26 descripteurs + spectre + modèle. Mis en cache par (fichier, vitesse, modèle)."""
    return diagnostiquer_fichier(donnees.modele(), io.BytesIO(contenu), vitesse, variante)


@st.cache_data(show_spinner=False, max_entries=60)
def _vitesse_auto(contenu):
    """Vitesse reconnue à partir du signal : (vitesse, confiance), ou None si le fichier est illisible."""
    m = donnees.modele_vitesse()
    if m is None:
        return None
    try:
        desc, _ = extraire_depuis_signal(lire_signal(io.BytesIO(contenu)))
    except SignalInvalide:
        return None
    return reco_vitesse.estimer(m, desc)


CSS = """<style>
[data-testid="stColumn"]:has(.st-key-diag_gauche){position:sticky;top:5rem;align-self:flex-start;z-index:2}
@media (max-width:900px){[data-testid="stColumn"]:has(.st-key-diag_gauche){position:static}}
</style>"""
def _x(rapport):
    return "×" + nombre(rapport, 1 if rapport < 100 else 0)


def _lecture(r, variante):
    """Encadré « Comment lire ce résultat » : une phrase en clair puis ce que veulent dire score, seuil et rapport."""
    rapport = r["score"] / r["seuil"]
    etat, prox = r["etat_estime"], r["proximite"][r["etat_estime"]]
    top_desc, top_part = next(iter(r["contributions"].items()))
    d_accord = sum(c["alarme"] == r["alarme"] for c in r["comparaison"])
    if r["alarme"]:
        phrase = T(f"Ce signal s’écarte <b>{_x(rapport)}</b> plus que la limite tolérée pour un roulement sain : Ballguard conclut à un <b>défaut</b>. "
                   f"Le type le plus probable est <b>{nom_etat(etat).lower()}</b> ({prox * 100:.0f} %).",
                   f"This signal deviates <b>{_x(rapport)}</b> more than the limit allowed for a healthy bearing: Ballguard concludes there is a <b>fault</b>. "
                   f"The most likely type is <b>{nom_etat(etat).lower()}</b> ({prox:.0%}).")
        if variante != "cnn":
            phrase += T(f" Le descripteur qui pèse le plus dans l’alarme : <b>{nom_desc(top_desc, True)}</b> ({top_part * 100:.0f} % de l’erreur).",
                        f" The feature weighing most in the alarm: <b>{nom_desc(top_desc, True)}</b> ({top_part:.0%} of the error).")
    else:
        phrase = T(f"Le score reste <b>sous la limite</b> des roulements sains ({_x(rapport)} le seuil) : Ballguard juge ce roulement <b>sain</b>.",
                   f"The score stays <b>below the limit</b> of healthy bearings ({_x(rapport)} the threshold): Ballguard judges this bearing <b>healthy</b>.")
    phrase += T(f" {d_accord} méthode(s) sur {len(r['comparaison'])} donnent le même verdict.",
                f" {d_accord} of {len(r['comparaison'])} method(s) give the same verdict.")
    avert = ""
    if r["alarme"] == (etat == "SAFE"):
        avert = (f'<div class="avert">{T("Attention : la détection et le type estimé ne concordent pas. Le signal est proche de la limite, le résultat est à confirmer.", "Caution: detection and estimated type disagree. The signal is close to the limit; the result should be confirmed.")}</div>')
    quoi = T("le spectre", "the spectrum") if variante == "cnn" else T("les 26 descripteurs", "the 26 features")
    lecture(phrase, [("Score", T(f"Erreur de reconstruction. L’autoencodeur a appris à reproduire {quoi} de roulements sains : plus le signal s’en écarte, plus il le reconstruit mal et plus le score monte.", f"Reconstruction error. The autoencoder learnt to reproduce {quoi} of healthy bearings: the further the signal is from them, the worse the reconstruction and the higher the score.")),
                     (T("Seuil", "Threshold"), T("La limite qui déclenche l’alarme, fixée pour que 99 % des roulements sains de validation restent en dessous.", "The limit that triggers the alarm, set so that 99% of healthy validation bearings stay below it.")),
                     (T("Rapport", "Ratio"), T("Score divisé par le seuil. Au-dessus de ×1, c’est une alarme ; plus il est grand, plus le signal est loin d’un roulement sain.", "Score divided by the threshold. Above ×1 it is an alarm; the larger it is, the further the signal is from a healthy bearing."))],
            titre=T("Comment lire ce résultat", "How to read this result"), avertissement=avert)


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
    html(CSS)
    gauche, droite = st.columns([1, 2.6], gap="medium")

    with gauche.container(border=True, key="diag_gauche"):
        titre_carte(T("Signal à analyser", "Signal to analyse"))
        televerses = st.file_uploader(T("Fichiers Excel du banc de test (.xlsx, 25 600 points)", "Test-bench Excel files (.xlsx, 25,600 points)"),
                                      type=["xlsx"], accept_multiple_files=True, key="diag_up",
                                      help=T("Les signaux du banc dépendent de la vitesse de rotation : Ballguard la reconnaît automatiquement à partir du signal.",
                                             "Bench signals depend on the rotation speed: Ballguard recognises it automatically from the signal."))
        if televerses:
            auto = {f.name: _vitesse_auto(f.getvalue()) for f in televerses}
            choix_v = st.segmented_control(T("Vitesse de rotation", "Rotation speed"), ["auto", *VITESSES], default="auto", key="diag_vit",
                                           format_func=lambda v: "Auto" if v == "auto" else str(v)) or "auto"
            reconnues = {n: a for n, a in auto.items() if a}
            if reconnues:
                detail = " · ".join(sorted({str(a[0]) for a in reconnues.values()}))
                st.caption(T(f"Vitesse détectée : {detail}", f"Detected speed: {detail}") + (
                    f" ({T('confiance', 'confidence')} {min(a[1] for a in reconnues.values()):.0%})" if len(reconnues) == 1 else
                    T(f" ({len(reconnues)} fichiers)", f" ({len(reconnues)} files)")))
            if choix_v == "auto":
                sources = [(f.name, f.getvalue(), auto[f.name][0] if auto[f.name] else 15) for f in televerses]
            else:
                sources = [(f.name, f.getvalue(), int(choix_v)) for f in televerses]
                ecarts = [n for n, a in reconnues.items() if a[0] != int(choix_v)]
                if ecarts:
                    liste = ", ".join(f"{n} : {reconnues[n][0]}" for n in ecarts)
                    st.warning(T(f"La vitesse choisie ({choix_v}) ne correspond pas au signal ({liste}). Le type de défaut risque d’être faux : revenez sur « Auto ».",
                                 f"The chosen speed ({choix_v}) does not match the signal ({liste}). The fault type may be wrong: switch back to “Auto”."),
                               icon=":material/warning:")
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
                    resultats[nom] = (*_analyser(nom, contenu, v, variante), v)
                except (SignalInvalide, VitesseInconnue) as e:
                    erreurs[nom] = str(e)
        for nom, msg in erreurs.items():
            st.error(f"**{nom}** : {T('fichier refusé', 'file rejected')}. {msg}", icon=":material/error:")
        if not resultats:
            st.info(T("Aucun signal exploitable pour l’instant.", "No usable signal yet."))
            return

        if len(resultats) > 1:
            with st.container(border=True, key=ncle()):
                titre_carte(T("Résultats du lot", "Batch results"), T(f"{len(resultats)} signaux analysés", f"{len(resultats)} signals analysed"))
                table = pd.DataFrame([{T("fichier", "file"): n, T("verdict", "verdict"): T("Défaut", "Fault") if r["alarme"] else T("Sain", "Healthy"),
                                       T("état estimé", "estimated state"): nom_etat(r["etat_estime"]), "score": round(r["score"], 4),
                                       T("vitesse", "speed"): v, T("score / seuil", "score / threshold"): round(r["score"] / r["seuil"], 2)} for n, (r, _, _, v) in resultats.items()])
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.download_button(T("Télécharger le tableau (CSV)", "Download the table (CSV)"), table.to_csv(index=False).encode("utf-8-sig"),
                                   "ballguard_resultats.csv", "text/csv", icon=":material/download:")
            nom = st.selectbox(T("Signal à détailler", "Signal to detail"), list(resultats), key="diag_sel")
        else:
            nom = next(iter(resultats))
        r, sig, spectre, vitesse = resultats[nom]
        mauvais = r["alarme"]

        html(f'<div class="bg-verdict {"bad" if mauvais else "good"}"><div class="ic">{"!" if mauvais else "✓"}</div>'
             f'<div><div class="f">{nom}</div><div class="big">{T("Défaut détecté", "Fault detected") if mauvais else T("Roulement sain", "Healthy bearing")}</div>'
             f'<div style="margin-top:.4rem">{pastille_etat(r["etat_estime"])}{pastille(T("vitesse", "speed") + f" {vitesse}")}{pastille(noms[variante])}</div></div>'
             f'<div class="meta"><div>Score<b>{score_txt(r["score"])}</b></div><div>{T("Seuil", "Threshold")}<b>{nombre(r["seuil"], 3)}</b></div>'
             f'<div>{T("Rapport", "Ratio")}<b>×{nombre(r["score"] / r["seuil"], 1 if r["score"] / r["seuil"] < 100 else 0)}</b></div></div></div>')

        _lecture(r, variante)
        jauge(r["score"], r["seuil"])

        o1, o2, o3, o4 = st.tabs([T("Explication", "Explanation"), T("Comparaison des modèles", "Model comparison"),
                                  T("Signal et spectre", "Signal and spectrum"), T("Descripteurs et export", "Features and export")])
        with o1:
            st.caption(T("À gauche, les descripteurs qui portent l’erreur de reconstruction ; à droite, de quel groupe de signaux connus le signal est le plus proche.",
                         "Left: the features carrying the reconstruction error. Right: which group of known signals the signal is closest to."))
            a, b = st.columns(2)
            with a.container(border=True, key=ncle()):
                titre_carte(T("Pourquoi ce verdict ?", "Why this verdict?"), T("Part de l’erreur de reconstruction portée par chaque descripteur (autoencodeur MLP)", "Share of the reconstruction error carried by each feature (MLP autoencoder)"))
                if variante == "cnn":
                    st.caption(T("Le CNN lit le spectre entier : son erreur ne se décompose pas par descripteur. Voici l’explication du MLP-AE pour le même signal.",
                                 "The CNN reads the whole spectrum: its error does not split by feature. Here is the MLP-AE explanation for the same signal."))
                hors = [f for f, v in r["descripteurs"].items() if not r["plage_saine"][f][0] <= v <= r["plage_saine"][f][1]]
                barres([(nom_desc(f), v, f"{v:.0%}") for f, v in list(r["contributions"].items())[:7]])
                st.caption(T(f"{len(hors)} descripteur(s) sur 26 hors de la plage des roulements sains à cette vitesse.", f"{len(hors)} of 26 feature(s) outside the healthy-bearing range at this speed."))
            with b.container(border=True, key=ncle()):
                titre_carte(T("Type de défaut estimé", "Estimated fault type"), (T("Probabilité d’appartenance à chaque groupe (GMM)", "Membership probability for each group (GMM)") if variante == "gmm"
                            else T("Proximité du signal à chaque groupe de l’espace latent", "Closeness of the signal to each latent-space group")))
                barres([(nom_etat(c), r["proximite"][c], f"{r['proximite'][c]:.0%}", p[c]) for c in CLASSES], texte=True)

        with o2:
            st.caption(T("Les méthodes de référence (Isolation Forest, ACP) et les autoencodeurs jugent le même signal, chacun avec sa propre limite. "
                         "« × seuil » : combien de fois le score dépasse cette limite. Si toutes donnent le même verdict, le résultat est robuste.",
                         "Baselines (Isolation Forest, PCA) and autoencoders judge the same signal, each with its own limit. "
                         "“× threshold”: how many times the score exceeds that limit. If all agree, the result is robust."))
            lignes = "".join(
                f'<tr class="{"hl" if c["modele"] == ("CNN-AE spectral" if variante == "cnn" else "Autoencodeur") else ""}"><td>{T("Autoencodeur", "Autoencoder") if c["modele"] == "Autoencodeur" else c["modele"]}</td>'
                f'<td>{pastille(T("Alarme", "Alarm"), p["alarm"]) if c["alarme"] else pastille("Normal", p["ok"])}</td>'
                f'<td class="n">×{nombre(c["score"] / c["seuil"], 2 if c["score"] / c["seuil"] < 100 else 0)}</td></tr>' for c in r["comparaison"])
            with st.container(border=True, key=ncle()):
                html(f'<div class="bg-scroll"><table class="bg-table"><tr><th>{T("Modèle", "Model")}</th><th>Verdict</th><th class="n">{T("× seuil", "× threshold")}</th></tr>{lignes}</table></div>')

        with o3:
            st.caption(T(f"En gris, un roulement sain enregistré à la même vitesse ({vitesse}). Un défaut se voit par des chocs dans le signal et des pics dans le spectre.",
                         f"In grey, a healthy bearing recorded at the same speed ({vitesse}). A fault shows as shocks in the signal and peaks in the spectrum."))
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

        with o4:
            st.caption(T("Valeur de chaque descripteur comparée à la plage des roulements sains (1er à 99e percentile) à la même vitesse.",
                         "Each feature’s value compared with the healthy-bearing range (1st to 99th percentile) at the same speed."))
            st.dataframe(pd.DataFrame([{T("descripteur", "feature"): nom_desc(f, True), T("code", "code"): f, T("valeur", "value"): v, T("plage saine (min)", "healthy range (min)"): r["plage_saine"][f][0],
                                        T("plage saine (max)", "healthy range (max)"): r["plage_saine"][f][1],
                                        T("dans la plage", "in range"): r["plage_saine"][f][0] <= v <= r["plage_saine"][f][1]} for f, v in r["descripteurs"].items()]),
                         hide_index=True, use_container_width=True, height=320)
            st.download_button(T("Télécharger le rapport du diagnostic (.txt)", "Download the diagnosis report (.txt)"), _rapport(nom, vitesse, r).encode("utf-8"),
                               f"ballguard_{nom.rsplit('.', 1)[0]}.txt", "text/plain", icon=":material/description:")
    pied()

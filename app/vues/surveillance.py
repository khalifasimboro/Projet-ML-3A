"""Surveillance en continu : on rejoue des acquisitions réelles dans l'ordre d'enregistrement."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ballguard.config import CLASSES, VITESSES
from ballguard.descripteurs import frequences_spectre
from ui import donnees
from ui.composants import ncle
from ui.composants import entete, html, kpi, pastille, pied, score_txt, titre_carte, tracer
from ui.i18n import T, nom_desc, nom_etat, nombre
from ui.theme import P

SUIVIS = ["rms_vb", "kurtosis_vb", "crete_vb", "pkpk_vb", "bande4_vb", "bande6_vb", "rms_v", "kurtosis_v"]
N_SAIN, N_DEFAUT = 50, 50


@st.cache_data(show_spinner=False, max_entries=20)
def _sequence(defaut, vitesse):
    """Acquisitions réelles dans l'ordre : une série saine, puis (éventuellement) une série en défaut."""
    bd = donnees.base()
    sain = bd[(bd.classe == "SAFE") & (bd.vitesse == vitesse)].sort_values("numero")
    if defaut == "SAFE":
        return sain.head(N_SAIN + N_DEFAUT).reset_index()
    panne = bd[(bd.classe == defaut) & (bd.vitesse == vitesse)].sort_values("numero")
    return pd.concat([sain.head(N_SAIN), panne.head(N_DEFAUT)]).reset_index()


def _carte(nom, hist, plage):
    """Carte d'un descripteur : valeur, mini-courbe, et bulle de détail au survol."""
    p, v, (lo, hi) = P(), hist[-1], plage
    hors = not lo <= v <= hi
    bas, haut = min(lo, min(hist)), max(hi, max(hist))
    Y = lambda x: 27 - (x - bas) / ((haut - bas) or 1) * 24
    q = hist[-60:]
    pts = " ".join(f"{i / max(len(q) - 1, 1) * 100:.1f},{Y(x):.1f}" for i, x in enumerate(q))
    courbe = (f'<rect x="0" y="{Y(hi):.1f}" width="100" height="{max(1, Y(lo) - Y(hi)):.1f}" fill="{p["ok"]}" opacity=".14"/>'
              f'<polyline fill="none" stroke="{p["alarm"] if hors else p["accent"]}" stroke-width="1.6" vector-effect="non-scaling-stroke" points="{pts}"/>')
    milieu = (lo + hi) / 2
    ecart = (v - milieu) / abs(milieu) * 100 if milieu else 0
    return (f'<div class="bg-fc {"out" if hors else ""}"><div class="nm">{nom}</div><div class="v">{nombre(v, 3)}</div>'
            f'<div class="tag">{T("hors plage", "out of range") if hors else "normal"}</div>'
            f'<svg viewBox="0 0 100 30" preserveAspectRatio="none">{courbe}</svg>'
            f'<div class="tip"><b>{nom}</b><div><span>{T("Valeur actuelle", "Current value")}</span><span>{nombre(v, 4)}</span></div>'
            f'<div><span>{T("Plage saine", "Healthy range")}</span><span>{nombre(lo, 3)} – {nombre(hi, 3)}</span></div>'
            f'<div><span>{T("Min / max de la série", "Series min / max")}</span><span>{nombre(min(hist), 3)} / {nombre(max(hist), 3)}</span></div>'
            f'<div><span>{T("Écart au sain", "Gap to healthy")}</span><span>{ecart:+.0f} %</span></div></div></div>')


def page():
    p, s = P(), st.session_state
    s.setdefault("sv_i", 0)
    s.setdefault("sv_run", False)
    entete(T("Analyse", "Analysis"), T("Surveillance en continu", "Live monitoring"),
           T("Rejouez des acquisitions réelles dans leur ordre d’enregistrement, comme si le capteur était branché sur la machine.",
             "Replay real acquisitions in recording order, as if the sensor were wired to the machine."))

    with st.container(border=True, key=ncle()):
        a, b, c, d = st.columns([2.2, 1, 1, 1.6], vertical_alignment="bottom")
        scenarios = [c_ for c_ in CLASSES if c_ != "SAFE"] + ["SAFE"]
        defaut = a.selectbox(T("Scénario", "Scenario"), scenarios, key="sv_sc",
                             format_func=lambda c_: T("Série saine uniquement", "Healthy series only") if c_ == "SAFE"
                             else T(f"Dégradation : sain puis {nom_etat(c_).lower()}", f"Degradation: healthy then {nom_etat(c_).lower()}"))
        vitesse = b.segmented_control(T("Vitesse", "Speed"), VITESSES, default=15, key="sv_vit") or 15
        cadence = c.segmented_control(T("Lecture", "Playback"), [1, 2, 4], default=2, format_func=lambda x: f"×{x}", key="sv_cad") or 2
        b1, b2 = d.columns(2)
        if b1.button(T("Pause", "Pause") if s.sv_run else T("Démarrer", "Start"), type="primary", use_container_width=True,
                     icon=":material/pause:" if s.sv_run else ":material/play_arrow:"):
            s.sv_run = not s.sv_run
            st.rerun()
        if b2.button(T("Réinitialiser", "Reset"), use_container_width=True, icon=":material/restart_alt:"):
            s.sv_i, s.sv_run = 0, False
            st.rerun()
    if s.get("sv_cle") != (defaut, vitesse):          # changement de scénario : on repart de zéro
        s.sv_cle, s.sv_i, s.sv_run = (defaut, vitesse), 0, False

    seq = _sequence(defaut, vitesse)
    seuil = donnees.modele()["detection_spectre"]["seuil"]          # modèle retenu : autoencodeur spectral
    ref = donnees.modele()["reference"][int(vitesse)]
    feats = donnees.modele()["features"]

    @st.fragment(run_every=(1.2 / cadence) if s.sv_run else None)
    def direct():
        s["_n"] = 1000
        if s.sv_run:
            s.sv_i = min(s.sv_i + 1, len(seq))
            if s.sv_i >= len(seq):
                s.sv_run = False
                st.rerun()
        i = s.sv_i
        vus = seq.iloc[:i]
        sc = vus.score_anomalie.to_numpy()
        alarme = sc > seuil
        dernier = vus.iloc[-1] if i else None
        mauvais = bool(i and alarme[-1])

        c = st.columns(4)
        kpi(c[0], T("État courant", "Current state"), (T("Alarme", "Alarm") if mauvais else "Normal") if i else "—",
            (nom_etat(dernier.etat_estime) if i else T("En attente de démarrage", "Waiting to start")), point=("red" if mauvais else "ok") if i else None)
        kpi(c[1], T("Acquisitions lues", "Acquisitions read"), str(i), T("1 acquisition = 1 seconde", "1 acquisition = 1 second"), f" / {len(seq)}")
        kpi(c[2], T("Alarmes", "Alarms"), str(int(alarme.sum())), T("Score au-dessus du seuil", "Score above threshold"), couleur=p["alarm"] if alarme.any() else None)
        kpi(c[3], T("Dernier score", "Last score"), score_txt(sc[-1]) if i else "—", f"{T('Seuil', 'Threshold')} {nombre(seuil, 3)}")
        st.write("")

        g, d = st.columns([2.3, 1])
        with g.container(border=True, key=ncle()):
            titre_carte(T("Score d’anomalie en direct", "Live anomaly score"))
            fig = go.Figure()
            fig.add_scatter(x=np.arange(1, i + 1), y=sc, mode="lines+markers", line=dict(color=p["accent"], width=2),
                            marker=dict(size=6, color=[p["alarm"] if a_ else p["ok"] for a_ in alarme]),
                            customdata=vus.fichier, hovertemplate="%{customdata}<br>score %{y:.3f}<extra></extra>")
            fig.add_hline(y=seuil, line=dict(color=p["fg"], dash="dash", width=1.2), annotation_text=T("seuil", "threshold"),
                          annotation_font_color=p["muted"])
            fig.add_hrect(y0=seuil, y1=10 ** 5, fillcolor=p["alarm"], opacity=0.06, line_width=0)
            fig.update_layout(xaxis=dict(title=T("numéro d’acquisition", "acquisition number"), range=[0, len(seq) + 1]),
                              yaxis=dict(title="score (log)", type="log", range=[-2.6, 5]))
            tracer(fig, 380, legende=False)
        with d.container(border=True, key=ncle()):
            titre_carte(T("Journal des événements", "Event log"))
            evts = []
            for k in range(i):
                avant = alarme[k - 1] if k else False
                if alarme[k] and not avant:
                    evts.append((k + 1, "a", T(f"Alarme : score {score_txt(sc[k])} · {nom_etat(vus.etat_estime.iloc[k])}",
                                               f"Alarm: score {score_txt(sc[k])} · {nom_etat(vus.etat_estime.iloc[k])}")))
                elif avant and not alarme[k]:
                    evts.append((k + 1, "o", T("Retour à la normale", "Back to normal")))
            corps = "".join(f'<div class="{t}"><time>n° {n}</time><span>{m}</span></div>' for n, t, m in reversed(evts)) or \
                f'<div><time>—</time><span>{T("Aucun événement pour l’instant.", "No event yet.")}</span></div>'
            html(f'<div class="bg-log">{corps}</div>')

        with st.container(border=True, key=ncle()):
            a_, b_ = st.columns([2, 1])
            a_.markdown(f'<p class="bg-h">{T("Acquisition en cours", "Current acquisition")}{" · " + dernier.fichier if i else ""}</p>'
                        f'<p class="bg-sub">{T("Spectre du signal et descripteurs calculés. Survolez une carte ou la courbe pour lire les valeurs.", "Signal spectrum and computed features. Hover a card or the curve to read the values.")}</p>',
                        unsafe_allow_html=True)
            b_.markdown(f'<div style="text-align:right">{pastille(T("dans la plage saine", "within healthy range"), p["ok"])}{pastille(T("hors plage", "out of range"), p["alarm"])}</div>', unsafe_allow_html=True)
            if not i:
                st.info(T("Cliquez sur Démarrer pour lire la première acquisition.", "Click Start to read the first acquisition."))
                return
            f = frequences_spectre() / 1000
            fig = go.Figure()
            fig.add_scatter(x=f, y=10 ** ref["spectre"][:256], name=T("référence saine", "healthy reference"), line=dict(color=p["muted"], width=1.1))
            fig.add_scatter(x=f, y=10 ** donnees.spectres()[int(dernier["index"])][:256], name=dernier.fichier,
                            line=dict(color=p["alarm"] if mauvais else p["accent"], width=1.3))
            fig.update_layout(xaxis_title=T("fréquence (kHz)", "frequency (kHz)"), yaxis=dict(type="log", title=T("puissance", "power")))
            tracer(fig, 250)
            cartes = "".join(_carte(nom_desc(n), vus[n].tolist(), (ref["p01"][feats.index(n)], ref["p99"][feats.index(n)])) for n in SUIVIS)
            html(f'<div class="bg-feats">{cartes}</div>')

    direct()
    pied()

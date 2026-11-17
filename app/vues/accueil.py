"""Accueil : l'essentiel en un écran (reprend la maquette validée)."""
import html as _html

import streamlit as st

from ui import donnees
from ui.composants import html, pied
from ui.i18n import T, nombre
from ui.theme import data_uri


def _svg(corps, couleur="currentColor", plein=False):
    remplissage = f'fill="{couleur}" stroke="none"' if plein else f'fill="none" stroke="{couleur}"'
    return (f'<svg viewBox="0 0 24 24" {remplissage} stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{corps}</svg>')


# icônes des indicateurs (à droite de chaque carte, comme sur la maquette)
I_BARRES = '<path d="M5 20v-6M10 20V9M15 20V4M20 20v-9"/>'
I_COCHE = '<circle cx="12" cy="12" r="9.5"/><path d="m7.5 12.3 3 3 6-6.3"/>'
I_CIBLE = '<circle cx="12" cy="12" r="9.5"/><circle cx="12" cy="12" r="4.5"/><path d="M12 12 19 5M16 4.5h3.5V8"/>'
I_RESEAU = '<circle cx="12" cy="4.5" r="2.2"/><circle cx="4.5" cy="18.5" r="2.2"/><circle cx="19.5" cy="18.5" r="2.2"/><path d="M10.9 6.5 5.6 16.5M13.1 6.5l5.3 10M6.8 18.5h10.4"/>'
I_POULS = '<path d="M2 12h4l2.5-6 4 13 3-9 2 2h4.5"/>'
# icônes des modules (pleines, orange)
M_ICONES = {
    "exploration": '<rect x="4" y="5" width="3" height="14" rx="1"/><rect x="10.5" y="5" width="3" height="14" rx="1"/><rect x="17" y="5" width="3" height="14" rx="1"/>',
    "diagnostic": '<path d="M12 2.5 21.5 12 12 21.5 2.5 12Z"/>',
    "surveillance": None,
    "latent": '<path fill-rule="evenodd" d="M12 2a10 10 0 1 1 0 20 10 10 0 0 1 0-20Zm0 3.2a6.8 6.8 0 1 0 0 13.6 6.8 6.8 0 0 0 0-13.6Zm0 3.4a3.4 3.4 0 1 1 0 6.8 3.4 3.4 0 0 1 0-6.8Z"/>',
    "performances": '<path d="M12 3.5 22 20.5H2Z"/>',
    "methodologie": '<path fill-rule="evenodd" d="M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm2.5 4.5v9h9v-9Z"/>',
}
VAGUE = '<path d="M2 14c2.5-6 5-6 7.5 0s5 6 7.5 0 3.5-4 5-2"/>'


def _css():
    return f"""<style>
.st-key-bghero{{position:relative;margin:-2.2rem -2.2rem 0;width:calc(100% + 4.4rem)!important;max-width:none!important;padding:2rem 2.2rem 2.4rem!important;border-radius:0 0 0 0;
  background:linear-gradient(90deg,var(--bg) 0%,var(--bg) 34%,rgba(0,0,0,0) 72%),linear-gradient(0deg,var(--bg) 0%,rgba(0,0,0,0) 45%),
  url("{data_uri('hero.jpg')}") right top/auto 118% no-repeat}}
.bg-hero h2{{font-family:var(--display);font-size:clamp(1.7rem,2.7vw,2.35rem);line-height:1.12;margin:0 0 .45rem;font-weight:600;padding:0;max-width:22ch}}
.bg-hero h2 em{{font-style:normal;color:var(--accent)}}
.bg-hero .salut{{font-size:.8rem;font-weight:600;letter-spacing:.06em;color:var(--accent);margin:0 0 .3rem}}
.bg-hero p.lead{{color:var(--muted);margin:0;font-size:.98rem}}
.bg-k{{display:flex;justify-content:space-between;gap:.8rem;align-items:flex-start;background:var(--surface);border:1px solid var(--line);
  border-radius:14px;padding:1.05rem 1.2rem;box-shadow:var(--shadow);height:100%}}
.bg-k .lab{{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}}
.bg-k .val{{font-family:var(--display);font-size:2rem;font-weight:600;line-height:1.2;font-variant-numeric:tabular-nums}}
.bg-k .val small{{font-size:1rem;color:var(--fg);font-weight:600;margin-left:.2rem}}
.bg-k .hint{{font-size:.8rem;color:var(--muted)}}
.bg-k svg{{width:38px;height:38px;flex:none;margin-top:.5rem}}
.st-key-bgpanneau{{background:var(--surface);border:1px solid var(--line)!important;border-radius:16px!important;box-shadow:var(--shadow);padding:1.1rem 1.2rem!important}}
.bg-ptitre{{display:flex;align-items:center;gap:.6rem;font-family:var(--display);font-weight:600;font-size:1.08rem;margin:0 0 .2rem}}
.bg-ptitre svg{{width:26px;height:26px}}
.bg-flux{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr;gap:.55rem;align-items:center}}
.bg-flux .st{{border:1px solid var(--line);background:var(--surface2);border-radius:10px;padding:.7rem .85rem;min-width:0}}
.bg-flux .st b{{display:block;font-family:var(--display);font-size:.9rem;font-weight:600}}
.bg-flux .st span{{font-size:.8rem;color:var(--muted)}}
.bg-flux .ar{{color:var(--fg);font-size:1rem}}
@media (max-width:900px){{.bg-flux{{grid-template-columns:1fr}}.bg-flux .ar{{display:none}}}}
[class*="st-key-bgmod"]{{background:var(--surface2);border:1px solid var(--line)!important;border-radius:14px!important;padding:.9rem 1.1rem!important;gap:.35rem}}
.bg-m .ic{{width:34px;height:34px;border-radius:50%;background:var(--accent-soft);display:grid;place-items:center;margin-bottom:.4rem}}
.bg-m .ic svg{{width:20px;height:20px}}
.bg-m b{{display:block;font-family:var(--display);font-size:1.02rem;font-weight:600}}
.bg-m{{padding-bottom:1.1rem}}
.bg-m p{{margin:.15rem 0 0!important;color:var(--muted);font-size:.86rem}}
[class*="st-key-bgmod"] [data-testid="stPageLink"] a{{display:inline-flex;padding:.35rem .9rem}}
</style>"""


def page():
    p = st.session_state
    art, pages = donnees.artefacts(), p["pages"]
    det = next(l for l in art["detection"]["table"] if l["modele"] == "AE (descripteurs)")
    n_sains = sum(det["confusion"][0])
    ac = "var(--accent)"
    html(_css())

    with st.container(key="bghero"):
        nom = _html.escape(p.get("nom", ""))
        st.markdown(f'<div class="bg-hero"><p class="salut">{T("Tableau de bord", "Dashboard")}</p>'
                    f'<h2>{T(f"Bonjour <em>{nom}</em>", f"Hello <em>{nom}</em>")}</h2></div>', unsafe_allow_html=True)

    cartes = [
        (T("Signaux analysés", "Signals analysed"), nombre(art["meta"]["n"], 0), "", T("5 états · 3 vitesses", "5 states · 3 speeds"), _svg(I_BARRES, ac)),
        (T("Défauts détectés", "Faults detected"), nombre(det["rappel"] * 100, 0), "%", T("Jeu de test", "Test set"), _svg(I_COCHE, "var(--ok)")),
        (T("Fausses alarmes", "False alarms"), nombre(det["fausses_alarmes"] * 100, 1), "%",
         T(f"{det['nb_fausses_alarmes']} signaux sains sur {n_sains}", f"{det['nb_fausses_alarmes']} healthy signals out of {n_sains}"), _svg(I_CIBLE, "var(--muted)")),
        (T("Accord clusters / états", "Cluster / state agreement"), nombre(art["clustering"]["corrige"]["autoencodeur"]["ARI"], 3), "", "ARI", _svg(I_RESEAU, "var(--INNER-RACE)")),
    ]
    for col, (lab, val, unite, hint, ic) in zip(st.columns(4), cartes):
        col.markdown(f'<div class="bg-k"><div><div class="lab">{lab}</div><div class="val">{val}{"<small>" + unite + "</small>" if unite else ""}</div>'
                 f'<div class="hint">{hint}</div></div>{ic}</div>', unsafe_allow_html=True)

    with st.container(key="bgpanneau", gap="medium"):
        etapes = [(T("Signal brut", "Raw signal"), T("Fichier Excel, 1 seconde", "Excel file, 1 second")),
                  (T("26 descripteurs", "26 features"), T("Temps et fréquence", "Time and frequency")),
                  (T("Autoencodeur", "Autoencoder"), T("Appris sur le sain", "Trained on healthy")),
                  (T("Détection", "Detection"), T("Sain ou défaut", "Healthy or fault")),
                  (T("Diagnostic", "Diagnosis"), T("Type de défaut", "Fault type"))]
        html(f'<div class="bg-ptitre">{_svg(I_POULS, ac)}{T("Du signal brut au diagnostic", "From raw signal to diagnosis")}</div>'
                '<div class="bg-flux">' + '<div class="ar">→</div>'.join(
                    f'<div class="st"><b>{t}</b><span>{s}</span></div>' for t, s in etapes) + "</div>")
        modules = [("exploration", T("Signaux, descripteurs, effet de la vitesse, corrélations.", "Signals, features, speed effect, correlations.")),
                   ("diagnostic", T("Déposez un fichier Excel : alarme, type de défaut et explication.", "Drop an Excel file: alarm, fault type and explanation.")),
                   ("surveillance", T("Rejouez une série d’acquisitions et suivez le score en direct.", "Replay an acquisition series and follow the score live.")),
                   ("latent", T("Visualisez comment l’autoencodeur organise les signaux.", "See how the autoencoder organises the signals.")),
                   ("performances", T("Autoencodeurs comparés aux modèles de référence.", "Autoencoders compared with baselines.")),
                   ("methodologie", T("Démarche, données, limites connues, équipe.", "Approach, data, known limits, team."))]
        for ligne in (modules[:3], modules[3:]):
            for col, (cle, texte) in zip(st.columns(3), ligne):
                with col.container(key=f"bgmod_{cle}"):
                    ic = _svg(VAGUE, ac) if M_ICONES[cle] is None else _svg(M_ICONES[cle], ac, plein=True)
                    html(f'<div class="bg-m"><div class="ic">{ic}</div><b>{pages[cle].title}</b><p>{texte}</p></div>')
                    st.page_link(pages[cle], label=T("Ouvrir", "Open"), icon=":material/arrow_forward:")
    pied()

"""Identité visuelle : deux thèmes construits sur les couleurs du logo Ballguard."""
import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

PALETTES = {
    "light": dict(base="light", bg="#F3F4F7", surface="#FFFFFF", surface2="#F6F7FA", line="#E0E4EC", fg="#101A2C",
                  muted="#5A667B", accent="#D9620B", accent_soft="rgba(228,110,20,.11)", ok="#128A56",
                  alarm="#D9293B", warn="#B07A08", shadow="0 1px 2px rgba(16,26,44,.06), 0 8px 24px rgba(16,26,44,.06)",
                  SAFE="#17996A", BALL="#2B7FD9", INNER_RACE="#8A5BE0", OUTER_RACE="#C9940F", COMBINED="#E03A5C"),
    "dark": dict(base="dark", bg="#0F1115", surface="#181B21", surface2="#20242C", line="#2D323D", fg="#F2F3F5",
                 muted="#9CA3B0", accent="#F27A1F", accent_soft="rgba(242,122,31,.15)", ok="#3DCB8A",
                 alarm="#FF5A6A", warn="#F2B93B", shadow="0 1px 0 rgba(255,255,255,.045) inset, 0 14px 34px rgba(0,0,0,.42)",
                 SAFE="#3DCB8A", BALL="#5AAEFF", INNER_RACE="#BC97FF", OUTER_RACE="#F4C653", COMBINED="#FF6B88"),
}
RAIL = dict(bg="#090A0D", line="#1D2028", fg="#D5D9E1", muted="#7C8492")
SERIE = ["#5AAEFF", "#FF6B88", "#3DCB8A", "#F4C653", "#BC97FF", "#3FD0D4", "#F58A2E", "#9AA7C0", "#E86FD0",
         "#7BD15A", "#5C7CFA", "#D98C5F", "#49B6A6", "#C2A7FF", "#FF9F9F"]


def P():
    """Palette du thème courant."""
    return PALETTES[st.session_state.get("theme", "light")]


def data_uri(nom):
    mime = "image/jpeg" if nom.endswith(".jpg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode((ASSETS / nom).read_bytes()).decode()


def appliquer():
    """Aligne le thème natif de Streamlit (widgets) sur la palette, puis injecte la feuille de style."""
    p = P()
    cibles = {"theme.base": p["base"], "theme.primaryColor": p["accent"], "theme.backgroundColor": p["bg"],
              "theme.secondaryBackgroundColor": p["surface2"], "theme.textColor": p["fg"]}
    change = False
    for cle, val in cibles.items():
        if st._config.get_option(cle) != val:
            st._config.set_option(cle, val)
            change = True
    if change and not st.session_state.get("_theme_rerun"):
        st.session_state["_theme_rerun"] = True
        st.rerun()
    st.session_state["_theme_rerun"] = False
    st.session_state["_n"] = 0
    st.markdown(_css(p), unsafe_allow_html=True)


def _css(p):
    v = ";".join(f"--{k.replace('_', '-')}:{val}" for k, val in p.items() if k != "base")
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{{v};--display:'Space Grotesk','Segoe UI',sans-serif;--body:'Inter','Segoe UI',sans-serif;--mono:'JetBrains Mono',Consolas,monospace}}
html,body,.stApp,[data-testid="stAppViewContainer"]{{background:var(--bg)!important;color:var(--fg);font-family:var(--body)}}
.stApp p,.stApp li,.stApp label,.stApp span,.stApp div{{font-family:var(--body)}}
.stApp [data-testid="stIconMaterial"],.stApp span[class*="material"]{{font-family:'Material Symbols Rounded'!important}}
[data-testid="stHeader"]{{background:transparent}}
.block-container{{padding:1.6rem 2.2rem 2.5rem!important;max-width:1720px}}
h1,h2,h3,h4{{font-family:var(--display)!important;letter-spacing:-.01em;color:var(--fg)}}
/* rail sombre dans les deux thèmes : il porte le logo ; compact pour tenir sans défiler */
[data-testid="stSidebar"]{{background:radial-gradient(120% 38% at 0% 100%,rgba(242,122,31,.30),rgba(242,122,31,.06) 45%,transparent 70%),{RAIL['bg']}!important;border-right:1px solid {RAIL['line']}}}
[data-testid="stSidebar"][data-collapsed="false"]{{width:262px!important;min-width:262px!important;max-width:262px!important}}
[data-testid="stSidebar"] *{{color:{RAIL['fg']}}}
[data-testid="stSidebarUserContent"]{{padding:.3rem 1rem 1rem!important}}
[data-testid="stSidebarNav"]{{padding-top:0}}
[data-testid="stSidebarNavItems"]{{padding:0 .6rem .2rem!important}}
[data-testid="stSidebarNavItems"] li{{margin:0}}
[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"]{{border-color:{RAIL['line']};margin:.3rem 0}}
[data-testid="stSidebarNavLink"]{{border-radius:8px;border:1px solid transparent;padding:.22rem .55rem!important;min-height:0;line-height:1.35;margin:1px 0}}
[data-testid="stSidebarNavLink"] span{{font-size:.82rem}}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"]{{font-size:1.05rem}}
[data-testid="stSidebarNavLink"]:hover{{background:#14161C}}
[data-testid="stSidebarNavLink"][aria-current="page"]{{background:linear-gradient(90deg,rgba(242,122,31,.22),rgba(242,122,31,.04));border-color:rgba(242,122,31,.38)}}
[data-testid="stSidebarNavLink"][aria-current="page"] span{{color:#fff;font-weight:600}}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"]{{color:#F58A2E}}
[data-testid="stNavSectionHeader"]{{margin:.55rem 0 .1rem!important;padding:0 .55rem!important;min-height:0}}
[data-testid="stSidebar"] header span,[data-testid="stNavSectionHeader"] span{{color:{RAIL['muted']}!important;font-size:.62rem;letter-spacing:.12em;text-transform:uppercase}}
[data-testid="stSidebar"] [data-testid="stLogo"],[data-testid="stSidebarHeader"] img{{height:4.1rem!important;max-width:100%;margin:.1rem auto 0}}
[data-testid="stSidebarHeader"]{{height:auto;padding:.8rem 1rem .1rem;justify-content:center;margin-bottom:0}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{{font-size:.7rem;color:{RAIL['muted']}!important}}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button{{min-height:1.7rem;padding:.1rem .42rem;font-size:.72rem}}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p{{font-size:.72rem}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{{gap:.6rem}}
[data-testid="stSidebar"] button[aria-checked="false"]{{background:#14161C!important;border-color:{RAIL['line']}!important}}
[data-testid="stSidebar"] button[aria-checked="false"] *{{color:{RAIL['fg']}!important}}
[data-testid="stSidebar"] button[aria-checked="true"]{{background:#F27A1F!important;border-color:#F27A1F!important}}
[data-testid="stSidebar"] button[aria-checked="true"] *{{color:#fff!important}}
.bg-model{{border:1px solid {RAIL['line']};border-radius:10px;padding:.55rem .75rem;font-size:.7rem;color:{RAIL['muted']}!important;background:rgba(15,17,22,.85);line-height:1.45}}
.bg-model b{{display:flex;align-items:center;gap:.45rem;font-weight:600;margin-bottom:.15rem}}
.bg-model span{{color:{RAIL['muted']}!important}}
/* cartes */
[class*="st-key-bgc"]{{background:var(--surface);border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:var(--shadow)}}
[data-testid="stMetric"]{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem;box-shadow:var(--shadow)}}
.stTabs [data-baseweb="tab-list"]{{gap:.25rem;border-bottom:1px solid var(--line)}}
.stTabs [data-baseweb="tab"]{{font-weight:600;color:var(--muted)}}
.stTabs [aria-selected="true"]{{color:var(--fg)}}
[data-testid="stFileUploaderDropzone"]{{background:var(--surface2);border:1.5px dashed var(--line);border-radius:12px}}
.stButton button,.stDownloadButton button{{border-radius:9px;font-weight:600}}
[data-testid="stPageLink"] a{{border:1px solid var(--line);border-radius:9px;background:var(--surface2);padding:.45rem .8rem}}
[data-testid="stPageLink"] a:hover{{border-color:var(--accent)}}
/* composants maison */
.bg-crumb{{font-size:.72rem;color:var(--muted);letter-spacing:.04em}}
.bg-title{{font-family:var(--display);font-size:1.5rem;font-weight:600;margin:0 0 .2rem}}
.bg-lead{{color:var(--muted);max-width:70ch;margin:0 0 .6rem}}
.bg-intro h2{{font-family:var(--display);font-size:clamp(1.5rem,2.6vw,2rem);line-height:1.15;margin:0 0 .35rem;font-weight:600;max-width:24ch;padding:0}}
.bg-intro h2 em{{font-style:normal;color:var(--accent)}}
.bg-h{{font-family:var(--display);font-size:1rem;font-weight:600;margin:0}}
.bg-sub{{color:var(--muted);font-size:.82rem;margin:.1rem 0 .4rem}}
.bg-kpi{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;box-shadow:var(--shadow);height:100%}}
.bg-kpi .lab{{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
.bg-kpi .val{{font-family:var(--display);font-size:1.9rem;font-weight:600;line-height:1.15;font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:.5rem}}
.bg-kpi .val small{{font-size:.95rem;color:var(--muted);font-weight:500}}
.bg-kpi .hint{{font-size:.76rem;color:var(--muted)}}
.bg-dot{{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(61,203,138,.2);display:inline-block;flex:none}}
.bg-dot.red{{background:var(--alarm);box-shadow:0 0 0 3px rgba(255,90,106,.2)}}
.bg-pill{{display:inline-flex;align-items:center;gap:.35rem;border-radius:999px;padding:.15rem .65rem;font-size:.76rem;font-weight:600;border:1px solid var(--line);background:var(--surface2);white-space:nowrap;margin:0 .3rem .3rem 0}}
.bg-pill i{{width:8px;height:8px;border-radius:50%;background:currentColor}}
.bg-pipe{{display:flex;gap:.5rem;flex-wrap:wrap;align-items:stretch}}
.bg-pipe .st{{flex:1 1 150px;border:1px solid var(--line);background:var(--surface2);border-radius:12px;padding:.75rem;min-width:0}}
.bg-pipe .st b{{display:block;font-family:var(--display);font-size:.86rem}}
.bg-pipe .st span{{font-size:.76rem;color:var(--muted)}}
.bg-pipe .ar{{align-self:center;color:var(--muted);font-size:1rem}}
.bg-mod .ic{{width:36px;height:36px;border-radius:10px;background:var(--accent-soft);color:var(--accent);display:grid;place-items:center;font-size:1.2rem;margin-bottom:.5rem}}
.bg-mod p{{margin:.15rem 0 .5rem;color:var(--muted);font-size:.82rem;min-height:2.4em}}
.bg-verdict{{border-radius:14px;padding:1.1rem 1.25rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;border:1px solid;margin-bottom:.4rem}}
.bg-verdict.bad{{background:linear-gradient(90deg,rgba(255,77,94,.16),rgba(255,77,94,.03));border-color:rgba(255,77,94,.45)}}
.bg-verdict.good{{background:linear-gradient(90deg,rgba(47,191,127,.16),rgba(47,191,127,.03));border-color:rgba(47,191,127,.45)}}
.bg-verdict .big{{font-family:var(--display);font-size:1.4rem;font-weight:700;line-height:1.15}}
.bg-verdict .ic{{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;flex:none;color:#fff;font-size:1.5rem;font-weight:700}}
.bg-verdict.bad .ic{{background:var(--alarm)}}.bg-verdict.good .ic{{background:var(--ok)}}
.bg-verdict .f{{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}}
.bg-verdict .meta{{margin-left:auto;display:flex;gap:1.1rem;flex-wrap:wrap}}
.bg-verdict .meta div{{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}
.bg-verdict .meta b{{display:block;font-family:var(--mono);font-size:.95rem;color:var(--fg);letter-spacing:0;text-transform:none}}
.bg-gauge{{position:relative;height:14px;border-radius:999px;background:linear-gradient(90deg,var(--ok) 0%,var(--ok) 22%,var(--warn) 29%,var(--alarm) 40%,var(--alarm) 100%);margin:2rem .6rem 1.9rem}}
.bg-gauge .th{{position:absolute;top:-8px;bottom:-8px;width:2px;background:var(--fg)}}
.bg-gauge .th span,.bg-gauge .mk span{{position:absolute;white-space:nowrap;font-size:.7rem;font-family:var(--mono);transform:translateX(-50%)}}
.bg-gauge .th span{{top:-18px;color:var(--muted)}}
.bg-gauge .mk{{position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:var(--surface);border:4px solid var(--fg);transform:translate(-50%,-50%);box-shadow:0 2px 8px rgba(0,0,0,.4)}}
.bg-gauge .mk span{{top:20px;font-weight:600;color:var(--fg)}}
.bg-bars{{display:flex;flex-direction:column;gap:.5rem;margin-bottom:.4rem}}
.bg-bars .b{{display:grid;line-height:1.3;min-height:1.3rem;grid-template-columns:124px minmax(0,1fr) 52px;gap:.6rem;align-items:center;font-size:.8rem}}
.bg-bars .b span:first-child{{font-family:var(--mono);font-size:.76rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bg-bars.txt .b span:first-child{{font-family:var(--body);font-size:.8rem}}
.bg-bars .tr{{height:9px;border-radius:99px;background:var(--surface2);overflow:hidden;border:1px solid var(--line)}}
.bg-bars .tr i{{display:block;height:100%;border-radius:99px;background:var(--accent)}}
.bg-bars .b span:last-child{{text-align:right;font-family:var(--mono);font-size:.76rem;color:var(--muted)}}
.bg-table{{border-collapse:collapse;width:100%;font-size:.82rem}}
.bg-table th,.bg-table td{{padding:.55rem .6rem;text-align:left;border-bottom:1px solid var(--line)}}
.bg-table th{{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}}
.bg-table td.n,.bg-table th.n{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.78rem}}
.bg-table tr.hl td{{background:var(--accent-soft)}}
.bg-scroll{{overflow-x:auto}}
.bg-log{{font-family:var(--mono);font-size:.76rem;display:flex;flex-direction:column;gap:.4rem;max-height:330px;overflow:auto}}
.bg-log div{{display:grid;grid-template-columns:54px minmax(0,1fr);gap:.5rem;padding:.45rem .55rem;border-radius:8px;background:var(--surface2);border-left:3px solid var(--line)}}
.bg-log div.a{{border-left-color:var(--alarm)}}.bg-log div.o{{border-left-color:var(--ok)}}
.bg-log time{{color:var(--muted)}}
.bg-feats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem}}
.bg-fc{{position:relative;border:1px solid var(--line);background:var(--surface2);border-radius:12px;padding:.65rem .75rem .5rem;display:flex;flex-direction:column;gap:.1rem;transition:border-color .15s,transform .15s}}
.bg-fc:hover{{border-color:var(--accent);transform:translateY(-2px);z-index:4}}
.bg-fc .nm{{font-family:var(--mono);font-size:.7rem;color:var(--muted)}}
.bg-fc .v{{font-family:var(--display);font-size:1.3rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2}}
.bg-fc.out{{border-color:var(--alarm)}}.bg-fc.out .v{{color:var(--alarm)}}
.bg-fc .tag{{font-size:.62rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ok)}}
.bg-fc.out .tag{{color:var(--alarm)}}
.bg-fc svg{{width:100%;height:30px;display:block;margin-top:.1rem}}
.bg-fc .tip{{display:none;position:absolute;left:0;bottom:calc(100% + 8px);min-width:220px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.6rem .75rem;box-shadow:0 12px 30px rgba(0,0,0,.35);font-size:.76rem}}
.bg-fc:hover .tip{{display:block}}
.bg-fc .tip b{{display:block;font-family:var(--display);font-size:.82rem;margin-bottom:.35rem}}
.bg-fc .tip div{{display:flex;justify-content:space-between;gap:.9rem;color:var(--muted);padding:.1rem 0}}
.bg-fc .tip div span:last-child{{font-family:var(--mono);color:var(--fg)}}
.bg-steps .s{{display:grid;grid-template-columns:34px minmax(0,1fr);gap:.9rem;padding-bottom:1.1rem;position:relative}}
.bg-steps .s:not(:last-child)::before{{content:"";position:absolute;left:16px;top:34px;bottom:0;width:2px;background:var(--line)}}
.bg-steps .n{{width:34px;height:34px;border-radius:50%;background:var(--accent-soft);color:var(--accent);display:grid;place-items:center;font-family:var(--display);font-weight:700}}
.bg-steps b{{font-family:var(--display);font-size:.92rem}}
.bg-steps p{{margin:.1rem 0 0;color:var(--muted);font-size:.82rem;max-width:70ch}}
.bg-people{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.6rem}}
.bg-people div{{border:1px solid var(--line);border-radius:10px;padding:.6rem .75rem;background:var(--surface2);font-weight:600;font-size:.82rem}}
.bg-people span{{display:block;font-weight:400;color:var(--muted);font-size:.76rem}}
.bg-note{{font-size:.78rem;color:var(--muted);border:1px dashed var(--line);border-radius:10px;padding:.6rem .75rem}}
.bg-foot{{color:var(--muted);font-size:.74rem;border-top:1px solid var(--line);padding-top:.8rem;margin-top:1.4rem;display:flex;justify-content:space-between;gap:.6rem;flex-wrap:wrap}}
</style>"""

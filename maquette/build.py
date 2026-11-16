import base64, pathlib
d = pathlib.Path(__file__).parent
b64 = lambda f: "data:image/png;base64," + base64.b64encode((d / f).read_bytes()).decode()
html = (d / "maquette.template.html").read_text(encoding="utf-8")
html = html.replace("__LOGO_FULL__", b64("logo_transparent.png")).replace("__LOGO_ICON__", b64("logo_icone.png"))
(d / "index.html").write_text(html, encoding="utf-8")
print("index.html", len(html) // 1024, "Ko")

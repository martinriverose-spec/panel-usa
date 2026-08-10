#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANEL USA — SWING AGRESIVO AUTOMATIZADO
Momentum swing 2-7 dias en valores USA liquidos de alto rango y precio bajo.

USO LOCAL:      python panel_usa.py          -> genera index.html
AUTOMATIZADO:   GitHub Actions lo ejecuta cada noche tras el cierre de NY
                y publica el panel en una URL que abres desde el movil.

ESTRATEGIA (racional):
  Momentum continuation: comprar fuerza probada, no anticipar giros.
  1. TENDENCIA   cierre > MA200 y > MA50 (solo se opera a favor de la marea)
  2. UBICACION   a <=6% del maximo de 20 sesiones (la fuerza reciente lidera)
  3. RANGO       ADR >= 4%: el valor se mueve lo suficiente para pagar +10-20%
  4. LIQUIDEZ    >= 25 M$/dia y precio 3-45 $ (capital pequeno compra varias acciones)
  5. EARNINGS    sin resultados en las proximas 8 sesiones (evento binario prohibido)
  6. CATALIZADOR titulares recientes en el panel; el juicio final es humano
  Disparador: buy-stop sobre el maximo de la ultima vela cerrada.
  Stop: minimo de ayer o 1,2 ATR (el mas lejano). Objetivos: +2 ATR y +3,5 ATR.
  Gestion: mitad fuera en OBJ1, stop a la entrada, resto a OBJ2. 5 sesiones max sin OBJ1.
"""

import warnings, html, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------- PARAMETROS ----------------
PRICE_MIN, PRICE_MAX = 3.0, 45.0     # rango de precio (capital pequeno)
ADR_MIN        = 4.0                 # % rango diario medio minimo
DVOL_MIN       = 25.0                # M$ volumen diario medio minimo
DIST_HI_MAX    = 6.0                 # % max por debajo del maximo de 20 sesiones
RISK_MAX_PCT   = 9.0                 # % maximo entre entrada y stop
EARNINGS_GAP   = 8                   # sesiones minimas hasta earnings
MAX_FINALISTS  = 3
NEWS_PER_STOCK = 3

UNIVERSE = [
 'PATH','SOFI','HOOD','AFRM','UPST','MARA','RIOT','CLSK','IONQ','RGTI','QBTS',
 'SOUN','BBAI','CVNA','DKNG','RBLX','U','ROKU','NU','GRAB','CPNG','RIVN','LCID',
 'NIO','XPEV','ENPH','RUN','PLUG','BE','CHPT','MP','AA','CLF','OXY','DVN','AR',
 'RRC','SM','CELH','ELF','ONON','ANF','URBN','W','CHWY','ETSY','LYFT','ZM','TWLO',
 'OKTA','S','PATH','AI','GTLB','DOCN','FSLY','MU','ON','WOLF','LSCC','SMR','OKLO',
 'NRG','TLN','HIMS','TDOC','OSCR','ALHC','NTRA','TWST','CRSP','NTLA','BEAM','RXRX',
 'TEM','KTOS','AVAV','RKLB','LUNR','ASTS','PL','BKSY','JOBY','ACHR','BILL','ESTC',
 'PTON','OPEN','LAZR','DNA','SPCE','GPRO','PSNY','FUBO','WULF','IREN','HUT','BTDR',
 'APLD','CORZ','SDGR','VKTX','ARWR','MDGL','IOVA','SRPT','RARE','FOLD','KRYS',
]

# ---------------- MOTOR ----------------

def _flat(df):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def scan():
    tickers = sorted(set(UNIVERSE))
    data = yf.download(tickers, period='1y', interval='1d', progress=False,
                       auto_adjust=True, group_by='ticker', threads=True)
    rows = []
    for tk in tickers:
        try:
            d = data[tk].dropna(subset=['Close'])
            if len(d) < 210:
                continue
            c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
            px, ph, pl = float(c.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1])
            ma50 = float(c.rolling(50).mean().iloc[-1])
            ma200 = float(c.rolling(200).mean().iloc[-1])
            hi20 = float(h.tail(20).max())
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            adr = float(((h - l) / c).tail(20).mean() * 100)
            dvol = float((c * v).tail(20).mean()) / 1e6
            rvol = float(v.iloc[-1]) / float(v.tail(20).mean())
            entry = ph * 1.001
            stop = max(pl, entry - 1.2 * atr)
            if entry - stop < 0.8 * atr:
                stop = entry - 1.0 * atr
            risk = entry - stop
            rows.append(dict(
                tk=tk, px=px, entry=entry, stop=stop,
                riskpct=risk / entry * 100,
                t1=entry + 2 * atr, t2=entry + 3.5 * atr,
                t1pct=2 * atr / entry * 100, t2pct=3.5 * atr / entry * 100,
                rr2=3.5 * atr / risk,
                adr=adr, dvol=dvol, rvol=rvol,
                chg20=(px / float(c.iloc[-21]) - 1) * 100,
                dist_hi=(hi20 / px - 1) * 100,
                vs50=(px / ma50 - 1) * 100, vs200=(px / ma200 - 1) * 100,
                fecha=str(d.index[-1].date())))
        except Exception:
            continue
    return pd.DataFrame(rows)


def gates(df):
    """Compuertas secuenciales. Devuelve (finalistas, descartes_con_motivo)."""
    out, dis = [], []
    for _, r in df.iterrows():
        if not (PRICE_MIN <= r.px <= PRICE_MAX):
            continue                                    # fuera de universo util, ni se lista
        if r.vs200 <= 0 or r.vs50 <= 0:
            dis.append((r.tk, 'Sin tendencia (bajo MA50/MA200)')); continue
        if r.adr < ADR_MIN:
            dis.append((r.tk, f'ADR {r.adr:.1f}% — no paga el objetivo')); continue
        if r.dvol < DVOL_MIN:
            dis.append((r.tk, f'Liquidez {r.dvol:.0f} M$/dia')); continue
        if r.dist_hi > DIST_HI_MAX:
            dis.append((r.tk, f'A {r.dist_hi:.1f}% del maximo: lejos del liderazgo')); continue
        if r.riskpct > RISK_MAX_PCT:
            dis.append((r.tk, f'Stop a {r.riskpct:.1f}%: riesgo excesivo')); continue
        out.append(r)
    out.sort(key=lambda r: r.dist_hi)
    return out, dis


def earnings_gate(cands):
    """Elimina candidatos con earnings dentro de la ventana. Devuelve (ok, fuera)."""
    hoy = datetime.now(timezone.utc).date()
    ok, out = [], []
    for r in cands:
        try:
            cal = yf.Ticker(r.tk).calendar
            eds = cal.get('Earnings Date') if isinstance(cal, dict) else None
            ed = eds[0] if eds else None
        except Exception:
            ed = None
        if ed is not None:
            dias = np.busday_count(hoy, ed)
            if dias <= EARNINGS_GAP:
                out.append((r.tk, f'Earnings {ed.strftime("%d/%m")} — a {dias} sesiones'))
                continue
            r['earn'] = ed.strftime('%d/%m')
        else:
            r['earn'] = 'sin fecha — VERIFICAR'
        ok.append(r)
    return ok, out


def get_news(tk):
    """Titulares recientes del valor. Datos reales de Yahoo; sin fecha no se muestra."""
    items = []
    try:
        for it in (yf.Ticker(tk).news or [])[:NEWS_PER_STOCK * 2]:
            c = it.get('content', it)
            title = c.get('title', '')
            date = str(c.get('pubDate', c.get('providerPublishTime', '')))[:10]
            prov = c.get('provider', {})
            prov = prov.get('displayName', '') if isinstance(prov, dict) else str(prov)
            link = c.get('canonicalUrl', {})
            link = link.get('url', '') if isinstance(link, dict) else c.get('link', '')
            if title and date:
                items.append(dict(t=title, d=date, p=prov, u=link))
            if len(items) >= NEWS_PER_STOCK:
                break
    except Exception:
        pass
    return items


# ---------------- HTML ----------------

def esc(s):
    return html.escape(str(s), quote=True)


def build_html(finalists, dis_tech, dis_earn, fecha_datos, repo_url=''):
    gen_dt = datetime.now(timezone.utc)
    gen = gen_dt.strftime('%d/%m/%Y %H:%M UTC')
    gen_iso = gen_dt.isoformat()
    actions_url = (repo_url.rstrip('/') + '/actions') if repo_url else ''

    def card(r, news, main):
        nh = ''.join(
            f"<a class='nw' href='{esc(n['u'])}' target='_blank' rel='noopener'>"
            f"<span class='nd'>{esc(n['d'])} · {esc(n['p'])}</span>{esc(n['t'])}</a>"
            for n in news) or "<div class='nw nd'>Sin titulares recientes en la fuente. Busca el valor antes de operar.</div>"
        cls = 'pick' if main else 'alt'
        badge = 'PRINCIPAL' if main else 'ALTERNATIVA'
        return f"""
<div class="{cls}" data-e="{r.entry:.2f}" data-s="{r.stop:.2f}" data-t2="{r.t2:.2f}" data-tk="{r.tk}">
  <div class="ph"><div class="t">{r.tk}<em>{r.px:.2f} $ · earnings {esc(r.get('earn','?'))}</em></div>
    <div class="b">{badge}</div></div>
  <div class="trg"><div class="lab">BUY-STOP — SI NO TOCA ESTE PRECIO, NO HAY OPERACION</div>
    <div class="px">{r.entry:.2f} $</div></div>
  <div class="lv">
    <div class="lc s"><div class="n">STOP</div><div class="p">{r.stop:.2f}</div><div class="x">-{r.riskpct:.1f}%</div></div>
    <div class="lc t"><div class="n">OBJ 1 (2 ATR)</div><div class="p">{r.t1:.2f}</div><div class="x">+{r.t1pct:.1f}%</div></div>
    <div class="lc t"><div class="n">OBJ 2 (3,5 ATR)</div><div class="p">{r.t2:.2f}</div><div class="x">+{r.t2pct:.1f}% · R:R {r.rr2:.1f}</div></div>
  </div>
  <div class="det">
    <div class="rw"><span class="kk">RACIONAL</span><span class="v">Lider de momentum: a {r.dist_hi:.1f}% de su maximo de 20 sesiones,
      +{r.chg20:.0f}% en el mes, +{r.vs200:.0f}% sobre MA200. ADR {r.adr:.1f}% — el objetivo de +{r.t2pct:.0f}% son 3,5 dias tipicos de rango.
      RVOL {r.rvol:.2f} · {r.dvol:.0f} M$/dia.</span></div>
    <div class="rw"><span class="kk">GESTION</span><span class="v">Mitad fuera en {r.t1:.2f}, stop a {r.entry:.2f}. Resto a {r.t2:.2f}.
      Cierra si en 5 sesiones no toco OBJ1. Cancela la orden si abre con hueco &gt;+4%.</span></div>
    <div class="rw news"><span class="kk">TITULARES</span><span class="v">{nh}</span></div>
  </div>
</div>"""

    finals_html = ''.join(card(r, get_news(r.tk), i == 0) for i, r in enumerate(finalists[:MAX_FINALISTS]))
    if not finalists:
        finals_html = ("<div class='empty'><div class='big'>NO OPERAR</div>"
                       "Ningun valor supera las compuertas hoy. Un dia sin operacion es un resultado valido: "
                       "el capital que no arriesgas en un mal dia es el que compone en los buenos.</div>")

    earn_rows = ''.join(f"<div class='dr'><span class='t'>{esc(t)}</span><span class='w'>{esc(m)}</span></div>"
                        for t, m in dis_earn)
    tech_rows = ''.join(f"<div class='dr'><span class='t'>{esc(t)}</span><span class='w'>{esc(m)}</span></div>"
                        for t, m in dis_tech[:10])

    opts = ''.join(f"<option value='{r.entry:.2f}|{r.stop:.2f}|{r.t2:.2f}'>{r.tk}</option>"
                   for r in finalists[:MAX_FINALISTS])

    return """<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Panel USA</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#04070C;--card:#0B1119;--card2:#101826;--line:#1E2A3C;--mint:#00FFB2;
--amber:#FFC400;--red:#FF3355;--blue:#4DA6FF;--w:#fff;--g1:#C3CDD9;--g2:#8595A8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--w);font-family:Archivo,system-ui,sans-serif;font-size:15px;line-height:1.5;padding-bottom:46px}
.wrap{max-width:760px;margin:0 auto;padding:0 14px}
header{padding:20px 0 10px}h1{font-size:24px;font-weight:900;letter-spacing:-.02em}
h1 span{color:var(--mint)}.sub{color:var(--g2);font-size:11.5px;font-family:'JetBrains Mono',monospace;margin-top:4px}
.fresh{display:flex;align-items:center;gap:11px;background:var(--card);border:1px solid var(--line);
padding:11px 13px;margin-top:12px}
.fdot{width:12px;height:12px;border-radius:50%;flex:0 0 12px;background:var(--g2)}
.ftxt{flex:1;min-width:0}
.fmain{font-size:13px;font-weight:700}
.fsub{font-size:10.5px;color:var(--g2);font-family:'JetBrains Mono',monospace;margin-top:1px}
.fbtns{display:flex;gap:6px;flex-wrap:wrap}
.frb,.fab{font-size:10.5px;font-weight:700;letter-spacing:.04em;padding:8px 11px;border:1px solid var(--line);
background:var(--card2);color:var(--g1);cursor:pointer;font-family:Archivo;text-decoration:none;white-space:nowrap}
.frb:hover,.fab:hover{border-color:var(--blue);color:#fff}
.fresh.ok .fdot{background:var(--mint);box-shadow:0 0 8px var(--mint)}
.fresh.warn .fdot{background:var(--amber)}
.fresh.old .fdot{background:var(--red)}
@media(max-width:520px){.fresh{flex-wrap:wrap}.fbtns{width:100%;justify-content:stretch}
.frb,.fab{flex:1;text-align:center}}
.sh{display:flex;align-items:center;gap:8px;margin:24px 0 10px}
.sh .d{width:10px;height:10px;border-radius:50%}.sh h2{font-size:16px;font-weight:900}
.pick{background:var(--card);border:2px solid var(--mint);margin-bottom:14px}
.alt{background:var(--card);border:1px solid var(--line);margin-bottom:12px}
.ph{padding:12px 14px;background:var(--card2);border-bottom:1px solid var(--line);
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px}
.ph .t{font-size:22px;font-weight:900}.ph .t em{font-style:normal;font-size:11px;color:var(--g2);font-weight:400;margin-left:8px}
.pick .ph .b{font-size:9.5px;font-weight:700;letter-spacing:.08em;color:#000;background:var(--mint);padding:4px 9px}
.alt .ph .b{font-size:9.5px;font-weight:700;letter-spacing:.08em;color:var(--amber);border:1px solid var(--amber);padding:4px 9px}
.trg{padding:12px 14px;background:#071510;border-bottom:1px solid var(--line)}
.trg .lab{font-size:8.5px;font-weight:700;letter-spacing:.1em;color:var(--mint);margin-bottom:5px}
.trg .px{font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:700;color:var(--mint);line-height:1}
.lv{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid var(--line)}
.lc{padding:10px 4px;text-align:center;border-right:1px solid var(--line)}.lc:last-child{border-right:0}
.lc .n{font-size:8.5px;font-weight:700;color:var(--g2);letter-spacing:.05em}
.lc .p{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;margin-top:3px}
.lc .x{font-size:9px;color:var(--g2);margin-top:2px;font-family:'JetBrains Mono',monospace}
.lc.s .p{color:var(--red)}.lc.t .p{color:var(--mint)}
.det{padding:12px 14px}
.rw{display:block;margin-bottom:10px;font-size:12.5px}.rw:last-child{margin-bottom:0}
.rw .kk{display:block;font-size:9px;font-weight:700;letter-spacing:.07em;color:var(--blue);margin-bottom:3px}
.rw .v{color:var(--g1);line-height:1.55}
.nw{display:block;color:var(--g1);text-decoration:none;padding:7px 0;border-bottom:1px solid rgba(30,42,60,.6);font-size:12px;line-height:1.45}
.nw:last-child{border-bottom:0}.nw:hover{color:#fff}
.nd{display:block;font-size:9.5px;color:var(--g2);font-family:'JetBrains Mono',monospace;margin-bottom:2px}
.empty{background:var(--card);border:2px solid var(--red);padding:24px 16px;text-align:center;color:var(--g1);font-size:13px}
.empty .big{font-size:30px;font-weight:900;color:var(--red);margin-bottom:8px}
.calc{background:var(--card);border:2px solid var(--blue);padding:14px}
.calc h3{font-size:12px;color:var(--blue);font-weight:900;letter-spacing:.05em;margin-bottom:10px}
.cin{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.cf label{display:block;font-size:8.5px;font-weight:700;letter-spacing:.06em;color:var(--g2);margin-bottom:4px}
.cf input,.cf select{width:100%;background:#060B12;border:1px solid var(--line);color:#fff;padding:8px 9px;
font-family:'JetBrains Mono',monospace;font-size:14px;border-radius:0}
.cf input:focus,.cf select:focus{outline:2px solid var(--blue);outline-offset:-1px}
.cout{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}
.co{background:var(--card2);padding:9px 6px;text-align:center}
.co .k{font-size:7.5px;font-weight:700;letter-spacing:.06em;color:var(--g2)}
.co .v{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;margin-top:3px}
.co.risk .v{color:var(--red)}.co.gain .v{color:var(--mint)}
@media(max-width:520px){.cin{grid-template-columns:1fr}.cout{grid-template-columns:repeat(2,1fr)}}
.jrn{background:var(--card);border:1px solid var(--amber);padding:14px}
.jrn h3{font-size:12px;color:var(--amber);font-weight:900;letter-spacing:.05em;margin-bottom:10px}
.jin{display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr auto;gap:6px;margin-bottom:10px}
.jin input,.jin select{background:#060B12;border:1px solid var(--line);color:#fff;padding:7px 8px;
font-family:'JetBrains Mono',monospace;font-size:12.5px;width:100%;border-radius:0}
.jin button,.jx{background:var(--amber);color:#000;border:0;font-weight:700;font-size:11px;
padding:7px 12px;cursor:pointer;letter-spacing:.04em;font-family:Archivo}
.jx{background:transparent;color:var(--g2);border:1px solid var(--line);margin-top:8px}
@media(max-width:560px){.jin{grid-template-columns:1fr 1fr}}
table{width:100%;border-collapse:collapse;font-size:11.5px;font-family:'JetBrains Mono',monospace}
th{text-align:right;padding:6px;color:var(--g2);font-size:8.5px;letter-spacing:.05em;border-bottom:1px solid var(--line)}
td{padding:6px;text-align:right;border-bottom:1px solid rgba(30,42,60,.5)}
th:first-child,td:first-child{text-align:left}
.pos{color:var(--mint)}.neg{color:var(--red)}
.dead{background:var(--card);border:1px solid var(--red)}
.dr{display:grid;grid-template-columns:64px 1fr;gap:9px;padding:8px 13px;border-bottom:1px solid var(--line);font-size:12px}
.dr:last-child{border-bottom:0}.dr .t{font-weight:900;color:var(--red)}.dr .w{color:var(--g1)}
footer{margin-top:24px;padding-top:13px;border-top:1px solid var(--line);color:var(--g2);font-size:10.5px;line-height:1.65}
footer b{color:var(--g1)}
</style></head><body><div class="wrap">
<header><h1>Panel <span>USA</span> · Swing</h1>
<div class="sub">Estrategia: momentum 2-7 dias, objetivo 2-3,5 ATR</div></header>

<div class="fresh" id="fresh">
  <div class="fdot" id="fdot"></div>
  <div class="ftxt">
    <div class="fmain" id="fmain">Cierre NY __FECHA__</div>
    <div class="fsub">Panel generado __GEN__</div>
  </div>
  <div class="fbtns">
    <button class="frb" onclick="location.href=location.pathname+'?t='+Date.now()">RECARGAR</button>
    __ACTIONSBTN__
  </div>
</div>

<div class="sh"><span class="d" style="background:var(--mint)"></span><h2 style="color:var(--mint)">Ordenes de hoy</h2></div>
__FINALES__

<div class="sh"><span class="d" style="background:var(--blue)"></span><h2 style="color:var(--blue)">Tamano de posicion</h2></div>
<div class="calc"><h3>CUANTAS ACCIONES</h3>
<div class="cin">
<div class="cf"><label>CAPITAL ($)</label><input type="number" id="cap" value="500"></div>
<div class="cf"><label>RIESGO (%)</label><input type="number" id="rsk" value="3" step="0.5"></div>
<div class="cf"><label>VALOR</label><select id="sym">__OPTS__</select></div></div>
<div class="cout">
<div class="co"><div class="k">ACCIONES</div><div class="v" id="osh">-</div></div>
<div class="co"><div class="k">POSICION</div><div class="v" id="opos">-</div></div>
<div class="co risk"><div class="k">STOP</div><div class="v" id="orisk">-</div></div>
<div class="co gain"><div class="k">OBJ 2</div><div class="v" id="ogain">-</div></div></div></div>

<div class="sh"><span class="d" style="background:var(--amber)"></span><h2 style="color:var(--amber)">Registro de operaciones</h2></div>
<div class="jrn"><h3>APUNTA CADA TRADE — TOMADO O SALTADO</h3>
<div class="jin">
<input id="jtk" placeholder="TICKER">
<input id="jen" type="number" step="0.01" placeholder="Entrada">
<input id="jsl" type="number" step="0.01" placeholder="Stop">
<input id="jex" type="number" step="0.01" placeholder="Salida (vacio=abierto)">
<button onclick="jadd()">GUARDAR</button></div>
<div id="jtab"></div>
<button class="jx" onclick="jcsv()">EXPORTAR CSV</button>
<button class="jx" onclick="if(confirm('Borrar todo el registro?')){localStorage.removeItem('trades');jrender()}">BORRAR</button>
<p style="color:var(--g2);font-size:10px;margin-top:8px">Se guarda en este navegador. Exporta el CSV cada semana como copia de seguridad.
El registro es lo que convierte la estrategia en datos: con 30-50 trades sabras tu tasa de acierto real.</p></div>

<div class="sh"><span class="d" style="background:var(--red)"></span><h2 style="color:var(--red)">Eliminadas hoy</h2></div>
<div class="dead">__EARN__ __TECH__</div>

<footer>
<p><b>Compuertas:</b> precio 3-45 $ · tendencia (sobre MA50 y MA200) · a menos del 6% del maximo de 20 sesiones ·
ADR minimo 4% · liquidez minima 25 M$/dia · stop maximo 9% · sin earnings en 8 sesiones · titulares a la vista.</p>
<p><b>El juicio final es humano:</b> lee los titulares antes de armar cada orden. Si el catalizador es negativo
(dilucion, guia recortada, investigacion), no operes aunque el tecnico sea perfecto.</p>
<p><b>Expectativa honesta:</b> perfil agresivo, tasa de acierto estimada 35-45% sin validar. Rentable solo si cada
acierto paga ~3x cada fallo y ejecutas todos los stops. Fechas de earnings de Yahoo: verificalas en el IR de cada
empresa. Panel de analisis, no recomendacion de inversion.</p></footer></div>
<script>
function calc(){var cap=+document.getElementById('cap').value||0,rp=+document.getElementById('rsk').value||0;
var sel=document.getElementById('sym');if(!sel.value)return;
var p=sel.value.split('|').map(Number),e=p[0],s=p[1],t2=p[2];
var sh=cap*rp/100/(e-s);if(sh*e>cap)sh=cap/e;sh=Math.floor(sh*100)/100;
document.getElementById('osh').textContent=sh.toFixed(2);
document.getElementById('opos').textContent='$'+(sh*e).toFixed(0);
document.getElementById('orisk').textContent='-$'+(sh*(e-s)).toFixed(2);
document.getElementById('ogain').textContent='+$'+(sh*(t2-e)).toFixed(2);}
['cap','rsk','sym'].forEach(function(id){document.getElementById(id).addEventListener('input',calc)});calc();
function jget(){try{return JSON.parse(localStorage.getItem('trades')||'[]')}catch(e){return[]}}
function jadd(){var t=document.getElementById('jtk').value.trim().toUpperCase();
var en=+document.getElementById('jen').value,sl=+document.getElementById('jsl').value,
exv=document.getElementById('jex').value,ex=exv===''?null:+exv;
if(!t||!en||!sl){alert('Ticker, entrada y stop son obligatorios');return}
var a=jget();a.push({d:new Date().toISOString().slice(0,10),t:t,e:en,s:sl,x:ex});
localStorage.setItem('trades',JSON.stringify(a));
['jtk','jen','jsl','jex'].forEach(function(i){document.getElementById(i).value=''});jrender()}
function jrender(){var a=jget(),el=document.getElementById('jtab');
if(!a.length){el.innerHTML='<p style="color:var(--g2);font-size:11px">Sin operaciones registradas.</p>';return}
var w=0,n=0,rs=0;var h='<table><tr><th>FECHA</th><th>TK</th><th>ENT</th><th>STOP</th><th>SAL</th><th>R</th></tr>';
a.slice().reverse().forEach(function(o){var r=o.x==null?null:(o.x-o.e)/(o.e-o.s);
if(r!=null){n++;rs+=r;if(r>0)w++}
h+='<tr><td>'+o.d+'</td><td>'+o.t+'</td><td>'+o.e.toFixed(2)+'</td><td>'+o.s.toFixed(2)+'</td><td>'+
(o.x==null?'abierto':o.x.toFixed(2))+'</td><td class="'+(r==null?'':r>0?'pos':'neg')+'">'+
(r==null?'-':r.toFixed(2))+'</td></tr>'});
h+='</table>';
if(n)h+='<p style="color:var(--g1);font-size:11px;margin-top:8px">Cerradas: '+n+' · Aciertos: '+w+' ('+
(100*w/n).toFixed(0)+'%) · R total: <b class="'+(rs>=0?'pos':'neg')+'">'+rs.toFixed(2)+'</b></p>';
el.innerHTML=h}
function jcsv(){var a=jget();if(!a.length)return;
var c='fecha,ticker,entrada,stop,salida,R\\n'+a.map(function(o){
var r=o.x==null?'':((o.x-o.e)/(o.e-o.s)).toFixed(3);
return[o.d,o.t,o.e,o.s,o.x==null?'':o.x,r].join(',')}).join('\\n');
var b=new Blob([c],{type:'text/csv'}),u=URL.createObjectURL(b),l=document.createElement('a');
l.href=u;l.download='trades.csv';l.click()}
jrender();
(function(){
  var gen=new Date("__GENISO__"), now=new Date(), hrs=(now-gen)/36e5;
  var el=document.getElementById('fresh'), sub=document.querySelector('.fsub');
  var cls='ok', msg='Datos al dia.';
  if(hrs>72){cls='old'; msg='Sin actualizar hace mas de 3 dias. Revisa que el workflow siga corriendo en Actions.';}
  else if(hrs>40){cls='warn'; msg='Puede faltar la sesion mas reciente. Si es antes de las 00:30 (Madrid) tras el cierre de NY, es normal.';}
  el.className='fresh '+cls;
  if(sub) sub.textContent+=' · '+msg;
})();
</script></body></html>""".replace('__FECHA__', fecha_datos).replace('__GEN__', gen).replace('__GENISO__', gen_iso) \
   .replace('__FINALES__', finals_html).replace('__OPTS__', opts) \
   .replace('__EARN__', earn_rows).replace('__TECH__', tech_rows) \
   .replace('__ACTIONSBTN__', f'<a class="fab" href="{actions_url}" target="_blank" rel="noopener">EJECUTAR AHORA</a>' if actions_url else '')


def main():
    print('Escaneando universo USA...')
    df = scan()
    if df.empty:
        print('Sin datos.'); return
    fecha = df.fecha.max()
    cands, dis_tech = gates(df)
    print(f'Velas hasta {fecha} · candidatos tecnicos: {len(cands)}')
    finalists, dis_earn = earnings_gate(cands[:8])
    print(f'Tras compuerta de earnings: {len(finalists)}')
    for r in finalists[:MAX_FINALISTS]:
        print(f"  {r.tk:5s} buy-stop {r.entry:7.2f}  stop {r.stop:7.2f} ({r.riskpct:4.1f}%)  "
              f"OBJ2 {r.t2:7.2f} (+{r.t2pct:4.1f}%)  R:R {r.rr2:.1f}  earnings {r.get('earn','?')}")
    import os
    repo = os.environ.get('GITHUB_REPOSITORY', '')  # p.ej. martinriverose-spec/panel-usa
    repo_url = f'https://github.com/{repo}' if repo else ''
    html_out = build_html(finalists, dis_tech, dis_earn, fecha, repo_url)
    Path('index.html').write_text(html_out, encoding='utf-8')
    print('Panel: index.html')


if __name__ == '__main__':
    main()

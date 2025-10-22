# ======================================================
# Dash Boursier — Pro v3
# Accueil IA synthèse + Global + Détail indices + Portefeuille (persistance) + 3+3 stratégiques
# ======================================================
import os, json
import requests, pandas as pd, numpy as np
import streamlit as st, altair as alt, yfinance as yf
from functools import lru_cache
from datetime import timedelta, timezone
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="Dash Boursier Pro v3", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
PF_CSV = os.path.join(DATA_DIR, "portefeuille.csv")
CFG_JSON = os.path.join(DATA_DIR, "config.json")

st.markdown('''
<style>
.pos {color:#0b8f3a; font-weight:600;}
.neg {color:#d5353a; font-weight:600;}
.neu {color:#1e88e5; font-weight:600;}
.chip {background:#f5f7fb; border:1px solid #e6e9f2; border-radius:12px; padding:4px 10px; margin-right:6px; font-size:0.85rem;}
.small {font-size:0.9rem; color:#555;}
.dataframe td, .dataframe th { font-size: 0.92rem; }
</style>
''', unsafe_allow_html=True)

PARIS_TZ = timezone(timedelta(hours=2))

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
try:
    sia = SentimentIntensityAnalyzer()
except Exception:
    sia = None

DEFAULT_CFG = {"mode":"Neutre","sentiment_threshold":0.15,"vol_threshold_pct":5.0,"score_buy":2,"score_sell":-2,"dividendes_enabled":False}

def load_config() -> dict:
    try:
        with open(CFG_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k,v in DEFAULT_CFG.items():
            cfg.setdefault(k,v)
        return cfg
    except Exception:
        return DEFAULT_CFG.copy()

def save_config(cfg: dict):
    try:
        with open(CFG_JSON, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_portfolio_df() -> pd.DataFrame:
    if os.path.exists(PF_CSV):
        try:
            df = pd.read_csv(PF_CSV)
            for col in ["Ticker","Quantité","PRU","Nom"]:
                if col not in df.columns: df[col] = np.nan
            df = df[["Ticker","Quantité","PRU","Nom"]]
            return df
        except Exception:
            pass
    return pd.DataFrame({"Ticker":["AIR.PA","MC.PA","VWCE.DE"],"Quantité":[10,5,8],"PRU":[140.0,700.0,110.0],"Nom":["Airbus","LVMH","ETF MSCI World"]})

def save_portfolio_df(df: pd.DataFrame):
    try: df.to_csv(PF_CSV, index=False)
    except Exception: pass

def parse_watchlist(text: str) -> list[str]:
    if not text: return []
    raw = [t.strip() for t in text.replace("\n", ",").replace(";", ",").split(",")]
    return [t for t in raw if t]

@st.cache_data(ttl=3600)
def _read_tables(url: str):
    headers={"User-Agent":"Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=15).text
    return pd.read_html(html)

def _extract_name_ticker(tables):
    table=None
    for df in tables:
        cols={str(c).lower() for c in df.columns}
        if ("company" in cols or "name" in cols) and ("ticker" in cols or "symbol" in cols):
            table=df.copy(); break
    if table is None: table=tables[0].copy()
    table.rename(columns={c:str(c).lower() for c in table.columns}, inplace=True)
    tcol=next((c for c in table.columns if "ticker" in c or "symbol" in c), table.columns[0])
    ncol=next((c for c in table.columns if "company" in c or "name" in c), table.columns[1])
    out=table[[tcol,ncol]].copy(); out.columns=["ticker","name"]
    out["ticker"]=out["ticker"].astype(str).str.strip()
    return out.dropna().drop_duplicates(subset=["ticker"])

@st.cache_data(ttl=3600)
def members_cac40():
    out=_extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/CAC_40"))
    out["ticker"]=out["ticker"].apply(lambda x: x if "." in x else f"{x}.PA"); return out

@st.cache_data(ttl=3600)
def members_dax40():
    out=_extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/DAX"))
    out["ticker"]=out["ticker"].apply(lambda x: x if "." in x else f"{x}.DE"); return out

@st.cache_data(ttl=3600)
def members_nasdaq100():
    return _extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/NASDAQ-100"))

@st.cache_data(ttl=3600)
def members_sp500():
    return _extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"))

@st.cache_data(ttl=3600)
def members_dowjones():
    return _extract_name_ticker(_read_tables("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"))

@st.cache_data(ttl=900)
def fetch_prices(tickers, days=120):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, period=f"{days}d", interval="1d", auto_adjust=False, group_by="ticker", threads=True, progress=False)
    frames=[]
    for t in tickers:
        try:
            if t in data:
                df=data[t].copy(); df["Ticker"]=t; frames.append(df)
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    out=pd.concat(frames); out.reset_index(inplace=True); return out

def compute_metrics(df):
    if df.empty: return df
    df = df.copy().sort_values(["Ticker","Date"])
    def change(x,n): return (x.iloc[-1]/x.iloc[-(n+1)])-1 if len(x)>n else np.nan
    pct1d  = df.groupby("Ticker")["Close"].apply(lambda x: (x.iloc[-1]/x.iloc[-2])-1 if len(x)>=2 else np.nan).rename("pct_1d")
    pct7d  = df.groupby("Ticker")["Close"].apply(lambda x: change(x,7)).rename("pct_7d")
    pct30d = df.groupby("Ticker")["Close"].apply(lambda x: change(x,22)).rename("pct_30d")
    def tr(g):
        pc=g["Close"].shift(1)
        return np.maximum(g["High"]-g["Low"], np.maximum(abs(g["High"]-pc), abs(g["Low"]-pc)))
    df["TR"]=df.groupby("Ticker").apply(tr).reset_index(level=0,drop=True)
    df["ATR14"]=df.groupby("Ticker")["TR"].transform(lambda s:s.rolling(14,min_periods=5).mean())
    df["MA20"]=df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(20,min_periods=5).mean())
    df["MA50"]=df.groupby("Ticker")["Close"].transform(lambda s:s.rolling(50,min_periods=10).mean())
    last=df.groupby("Ticker").tail(1)[["Ticker","Date","Close","ATR14","MA20","MA50"]]
    return (last.merge(pct1d,left_on="Ticker",right_index=True).merge(pct7d,left_on="Ticker",right_index=True).merge(pct30d,left_on="Ticker",right_index=True))

@lru_cache(maxsize=1024)
def google_news(q, lang="fr"):
    url=f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl={lang}-{lang.upper()}&gl={lang.upper()}&ceid={lang.upper()}:{lang.upper()}"
    try:
        xml=requests.get(url,timeout=10).text
        import xml.etree.ElementTree as ET
        root=ET.fromstring(xml)
        titles=[(i.findtext("title") or "") for i in root.iter("item")]
        return titles[:6]
    except Exception:
        return []

def news_summary(name, ticker, lang="fr"):
    titles = google_news(f"{name} {ticker}", lang) or google_news(name, lang)
    if not titles:
        return ("Pas d’actualité saillante — mouvement possiblement technique (flux, arbitrages, macro).", 0.0)
    scores=[]; KEY_POS=["résultats","bénéfice","guidance","relève","contrat","approbation","dividende","rachat","upgrade","partenariat"]
    KEY_NEG=["profit warning","avertissement","enquête","retard","rappel","amende","downgrade","abaisse","procès"]
    for t in titles:
        s=0
        if sia:
            try: s = sia.polarity_scores(t.lower())["compound"]
            except Exception: s=0
        if any(k in t.lower() for k in KEY_POS): s += 0.2
        if any(k in t.lower() for k in KEY_NEG): s -= 0.2
        scores.append(s)
    m=float(np.mean(scores)) if scores else 0.0
    if m>0.15: txt="Hausse soutenue par des nouvelles positives (résultats/contrats/relèvement)."
    elif m<-0.15: txt="Baisse liée à des nouvelles défavorables (abaissement, retard, pression sectorielle)."
    else: txt="Actualité mitigée/neutre : mouvement surtout technique (flux, rotation, macro)."
    return (txt, m)

def detailed_risk_analysis(row, sentiment_score):
    px = float(row.get("Close", np.nan))
    atr = float(row.get("ATR14", np.nan)) if not pd.isna(row.get("ATR14", np.nan)) else 0.02*px
    ma20 = float(row.get("MA20", px if not np.isnan(px) else 0))
    ma50 = float(row.get("MA50", px if not np.isnan(px) else 0))
    pct1 = float(row.get("pct_1d", 0) or 0); pct7 = float(row.get("pct_7d", 0) or 0); pct30 = float(row.get("pct_30d", 0) or 0)
    vol = (atr/px) if px else 0.03
    trend_up = (px >= ma20) + (px >= ma50)
    mom_up = sum([(pct1>0),(pct7>0),(pct30>0)])
    def bucket_risk(v):
        if v < 0.015: return "Faible"
        if v < 0.035: return "Moyen"
        return "Élevé"
    risk_ct = bucket_risk(vol)
    score_mid = trend_up + mom_up + (1 if sentiment_score>0.15 else -1 if sentiment_score<-0.15 else 0)
    score_long = mom_up + (2 if trend_up==2 else 0)
    buy = max(ma20, px - 0.5*atr); stop = buy - 1.5*atr; tgt  = buy + 3*atr
    ct = f"CT (1–4 sem.) : {risk_ct}, vol ~{vol*100:.1f}%."; mt = f"MT (3–6 mois) : {'Favorable' if score_mid>=3 else 'Neutre' if score_mid>=1 else 'Délicat'}."; lt = f"LT (6–12 mois) : {'Positif' if score_long>=3 else 'Incertain'}."
    return ct, mt, lt, buy, stop, tgt, risk_ct

def decision_signal(row, sentiment, pnl_pct=None, cfg:dict=None, context="market"):
    cfg = cfg or {}
    sent_th = float(cfg.get("sentiment_threshold", 0.15))
    vol_th  = float(cfg.get("vol_threshold_pct", 5.0))/100.0
    buy_sc  = int(cfg.get("score_buy", 2))
    sell_sc = int(cfg.get("score_sell", -2))
    px = float(row.get("Close", np.nan)); ma20 = row.get("MA20", np.nan); ma50 = row.get("MA50", np.nan); atr = row.get("ATR14", np.nan)
    trend_up = int(px>=ma20) + int(px>=ma50) if pd.notna(ma20) and pd.notna(ma50) and pd.notna(px) else 1
    vol = (atr/px) if (pd.notna(atr) and pd.notna(px) and px>0) else 0.025
    score = 0
    score += (1 if trend_up==2 else -1 if trend_up==0 else 0)
    score += (1 if sentiment>sent_th else -1 if sentiment<-sent_th else 0)
    if pnl_pct is not None: score += (1 if pnl_pct<0 else -1 if pnl_pct>15 else 0)
    if vol>vol_th: score -= 1
    if context == "portfolio":
        if score >= buy_sc:   return "🟢", "ACHETER / Renforcer", "Tendance + actus favorables, point d’entrée correct."
        if score <= sell_sc:  return "🔴", "VENDRE / Alléger", "Contexte fragile (tendance/actus/volatilité), prudence."
        return "🟡", "GARDER", "Neutre : conserver, attendre meilleur point."
    else:
        if score >= buy_sc:   return "🟢", "Acheter / Entrée favorable", "Signal haussier + contexte porteur."
        if score <= sell_sc:  return "🔴", "Éviter / Pas d’opportunité", "Risque élevé ou structure fragile."
        return "⚪", "Surveiller de près", "Changement de ton/tendance possible."

def colorize_table(df, value_col, sentiment_col=None, risk_col=None, decision_col=None):
    df_show = df.copy()
    if value_col in df_show:
        df_show["Variation %"] = df_show[value_col]*100 if df_show[value_col].max()<=1 else df_show[value_col]
    def color_var(v):
        if pd.isna(v): return ""
        if v > 0: return "background-color: #e8f5e9; color: #0b8f3a"
        if v < 0: return "background-color: #ffebee; color: #d5353a"
        return "background-color: #e8f0fe; color: #1e88e5"
    def color_sent(s):
        if pd.isna(s): return ""
        if s > 0.15: return "background-color:#e8f5e9; color:#0b8f3a"
        if s < -0.15: return "background-color:#ffebee; color:#d5353a"
        return "background-color:#e8f0fe; color:#1e88e5"
    def color_risk(r):
        r = str(r)
        if "Élevé" in r: return "background-color:#ffebee; color:#d5353a"
        if any(x in r for x in ["Faible","Favorable","Positif"]): return "background-color:#e8f5e9; color:#0b8f3a"
        return "background-color:#e8f0fe; color:#1e88e5"
    def color_dec(s):
        s=str(s)
        if "🟢" in s: return "background-color:#e8f5e9; color:#0b8f3a"
        if "🔴" in s: return "background-color:#ffebee; color:#d5353a"
        if "🟡" in s or "⚪" in s: return "background-color:#e8f0fe; color:#1e88e5"
        return ""
    sty = df_show.style
    if "Variation %" in df_show: sty = sty.applymap(color_var, subset=["Variation %"])
    if sentiment_col and sentiment_col in df_show: sty = sty.applymap(color_sent, subset=[sentiment_col])
    if risk_col and risk_col in df_show: sty = sty.applymap(color_risk, subset=[risk_col])
    if decision_col and decision_col in df_show: sty = sty.applymap(color_dec, subset=[decision_col])
    return sty

def bar_chart(df, val, title):
    if df.empty: st.info("Aucune donnée à afficher."); return
    d = df.copy()
    d["Name"] = d.get("name", d.get("Name", d.get("Ticker", ""))).fillna(d.get("Ticker","")).astype(str)
    d["pct"] = (d[val]*100) if d[val].max()<=1 else d[val]
    d["color"] = np.where(d["pct"] >= 0, "Hausses", "Baisses")
    chart = alt.Chart(d).mark_bar().encode(
            x=alt.X("Name:N", sort="-y", title="Société"),
            y=alt.Y("pct:Q", title="Variation (%)"),
            color=alt.Color("color:N", scale=alt.Scale(domain=["Hausses","Baisses"], range=["#0b8f3a","#d5353a"]), legend=None),
            tooltip=["Name","Ticker",alt.Tooltip("pct",format=".2f")]
        ).properties(title=title, height=320)
    st.altair_chart(chart, use_container_width=True)

def plot_90_with_levels(ticker, name, buy, stop, tgt):
    if not ticker: return
    data = yf.download(ticker, period="100d", interval="1d", auto_adjust=False, progress=False)
    if data.empty: st.info("Données 90j indisponibles."); return
    data = data.reset_index()
    base = alt.Chart(data).mark_line().encode(
        x=alt.X("Date:T", title=""),
        y=alt.Y("Close:Q", title="Cours"),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=".2f")]
    ).properties(title=f"{name} — 90 jours")
    rules = alt.Chart(pd.DataFrame({"y":[buy, tgt, stop], "label":["Achat ~","Cible ~","Stop ~"]})).mark_rule().encode(y="y:Q", tooltip=["label:N","y:Q"])
    st.altair_chart(base + rules, use_container_width=True)

UNIVERSES = ["CAC 40","DAX 40","NASDAQ 100","S&P 500","Dow Jones","LS Exchange"]

def build_universe(market:str, watchlist_text:str) -> pd.DataFrame:
    if market=="CAC 40": return members_cac40()
    if market=="DAX 40": return members_dax40()
    if market=="NASDAQ 100": return members_nasdaq100()
    if market=="S&P 500": return members_sp500()
    if market=="Dow Jones": return members_dowjones()
    if market=="LS Exchange":
        wl=parse_watchlist(watchlist_text)
        if not wl:
            st.warning("LS Exchange : ajoute une watchlist de tickers (sidebar)."); return pd.DataFrame()
        return pd.DataFrame({"ticker":wl, "name":wl})
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_all_markets(markets_and_watchlists: list[tuple[str,str]], days_hist=120):
    combined=[]
    for (mkt,wltext) in markets_and_watchlists:
        members = build_universe(mkt, wltext)
        if members.empty: continue
        px = fetch_prices(members["ticker"].tolist(), days=days_hist)
        if px.empty: continue
        px = px.merge(members, left_on="Ticker", right_on="ticker", how="left")
        metrics = compute_metrics(px).merge(members, left_on="Ticker", right_on="ticker", how="left")
        metrics["Indice"] = mkt
        combined.append(metrics)
    if not combined: return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)

def build_analysis_table(dfset: pd.DataFrame, value_col: str, cfg: dict, context="market") -> pd.DataFrame:
    rows=[]
    for _, r in dfset.iterrows():
        name = (r.get("name") or r.get("Name") or r.get("Ticker"))
        tick = r.get("Ticker","")
        px = float(r.get("Close", np.nan))
        txt, sc = news_summary(str(name), tick)
        ct, mt, lt, buy, stop, tgt, risk_ct = detailed_risk_analysis(r, sc)
        emoji, deci, why = decision_signal(r, sc, None, cfg=cfg, context=context)
        flash = f"Tendance: {('>MA20 & >MA50' if (pd.notna(r.get('MA20')) and pd.notna(r.get('MA50')) and px>=r.get('MA20') and px>=r.get('MA50')) else 'fragile')}. {txt}"
        rows.append({"Name": str(name), "Ticker": tick, "Cours": round(px,2),
            "Variation %": round((r.get(value_col,0)*100),2),
            "MA20": round(float(r.get("MA20", float("nan"))),2) if pd.notna(r.get("MA20", np.nan)) else np.nan,
            "MA50": round(float(r.get("MA50", float("nan"))),2) if pd.notna(r.get("MA50", np.nan)) else np.nan,
            "ATR14": round(float(r.get("ATR14", float("nan"))),2) if pd.notna(r.get("ATR14", np.nan)) else np.nan,
            "Sentiment": sc, "Risque CT": risk_ct,
            "Conseil": f"Achat~{buy:.2f} / Cible~{tgt:.2f} / Stop~{stop:.2f}",
            "Décision IA": f"{emoji} {deci}",
            "Analyse IA": flash})
    return pd.DataFrame(rows)

def pick_candidates(full_df: pd.DataFrame, value_col: str, cfg: dict):
    df = full_df.copy()
    df["Name"] = df.get("name", df.get("Name", df.get("Ticker",""))).astype(str)
    sentiments_local=[]
    for _, r in df.iterrows():
        nm=str(r["Name"]); _, sc = news_summary(nm, r.get("Ticker","")); sentiments_local.append(sc)
    df["sentiment"]=sentiments_local
    df["vol"]=(df["ATR14"]/df["Close"]).replace([np.inf,-np.inf],np.nan).fillna(0.03)
    df["trend_up"]=((df["Close"]>=df["MA20"]).astype(int)+(df["Close"]>=df["MA50"]).astype(int))
    df["score_fast"]=(df["vol"].rank(pct=True))+(df[value_col].rank(pct=True))+((df["sentiment"]>0).astype(int))
    high_fast=df.sort_values("score_fast",ascending=False).head(3)
    df["score_long"]=(1-df["vol"].rank(pct=True))+df["trend_up"]+((df["sentiment"]>=0).astype(int))+((df.get("pct_30d",0)>=0).astype(int))
    low_long=df.sort_values("score_long",ascending=False).head(3)
    return build_analysis_table(high_fast, value_col, cfg, context="market"), build_analysis_table(low_long, value_col, cfg, context="market")

cfg = load_config()
with st.sidebar:
    st.header("Réglages IA")
    mode = st.selectbox("Profil IA", ["Offensif","Neutre","Prudent"], index=["Offensif","Neutre","Prudent"].index(cfg.get("mode","Neutre")))
    if mode=="Offensif": defaults={"sentiment_threshold":0.10,"vol_threshold_pct":7.0,"score_buy":1,"score_sell":-3}
    elif mode=="Prudent": defaults={"sentiment_threshold":0.20,"vol_threshold_pct":3.0,"score_buy":3,"score_sell":-1}
    else: defaults={"sentiment_threshold":0.15,"vol_threshold_pct":5.0,"score_buy":2,"score_sell":-2}
    apply_profile = st.checkbox("Appliquer le profil automatiquement", value=True)
    if apply_profile: cfg.update({"mode":mode, **defaults})
    cfg["sentiment_threshold"] = st.slider("Sensibilité sentiment (IA)", 0.0, 0.5, float(cfg["sentiment_threshold"]), 0.01)
    cfg["vol_threshold_pct"]  = st.slider("Volatilité max tolérée (%)", 0.0, 15.0, float(cfg["vol_threshold_pct"]), 0.5)
    cfg["score_buy"]          = st.slider("Score minimal pour 'Acheter'", 0, 5, int(cfg["score_buy"]))
    cfg["score_sell"]         = st.slider("Score maximal pour 'Vendre'", -5, 0, int(cfg["score_sell"]))
    st.button("💾 Sauvegarder réglages", on_click=lambda: save_config(cfg))
    st.markdown("---")
    page_selector = st.radio("Page", ["🏠 Accueil IA", "🌍 Marché Global", "📊 Détail par Indice", "📁 Mon Portefeuille"], index=0)
    periode = st.radio("Période d’analyse", ["Jour","7 jours","30 jours"], index=0)
    show_atr_ma = st.toggle("Afficher MA20 / MA50 / ATR14", value=False)
    st.markdown("### 🎛 Watchlist (LS Exchange)")
    wl_ls   = st.text_area("Tickers LS Exchange (facultatif)", value="", height=80, placeholder="Ex: AIR.PA, ORA.PA, MC.PA")
    univers = st.selectbox("Indice (pour la vue Détail)", UNIVERSES, index=0)

with st.expander("📘 Comprendre les indicateurs (ATR / MA)", expanded=False):
    st.markdown("- **ATR14** : amplitude moyenne des variations sur 14 séances → **volatilité**.
- **MA20 / MA50** : moyennes mobiles 20/50 jours → **tendance**.
**Prix > MA20 & MA50** = tendance solide ; **Prix < MA20 & MA50** = faiblesse.")

days_hist = 60 if periode=="Jour" else (90 if periode=="7 jours" else 150)
value_col = {"Jour":"pct_1d","7 jours":"pct_7d","30 jours":"pct_30d"}[periode]
topN = 5
MARKETS_AND_WL = [("CAC 40",""),("DAX 40",""),("NASDAQ 100",""),("S&P 500",""),("Dow Jones",""),("LS Exchange", wl_ls)]

if page_selector == "🏠 Accueil IA":
    st.title("🏠 Accueil — Synthèse & Alertes IA")
    all_data = fetch_all_markets(MARKETS_AND_WL, days_hist=days_hist)
    pf_df = load_portfolio_df()
    pf_df = pf_df[pf_df["Ticker"].astype(str).str.len()>0]
    if all_data.empty: st.warning("Aucun univers disponible."); st.stop()
    valid = all_data.dropna(subset=[value_col]).copy()
    if valid.empty: st.warning("Pas de variations calculables."); st.stop()
    avg = valid[value_col].mean()*100; up=(valid[value_col]>0).sum(); down=(valid[value_col]<0).sum()
    sample_names = valid.sort_values(value_col, ascending=False).head(6).get("name", valid.get("Name", pd.Series(dtype=str))).fillna("").astype(str).tolist()
    sentiments=[news_summary(nm, nm)[1] for nm in sample_names if nm]
    avg_sent = float(np.mean(sentiments)) if sentiments else 0.0
    global_sent = "Marché positif (flux acheteurs, bonnes publications)." if avg_sent>0.15 else ("Marché sous pression (macro/news défavorables)." if avg_sent<-0.15 else "Tonalité neutre : mouvements techniques.")
    st.markdown(f"<div class='chip'>Résumé global ({periode})</div> <span class='{'pos' if avg>=0 else 'neg'}'>Variation moyenne : {avg:.2f}%</span> — <span class='pos'>{up} hausses</span> / <span class='neg'>{down} baisses</span><br><span class='small'>{global_sent}</span>", unsafe_allow_html=True)
    s = valid.sort_values(value_col, ascending=False); top = s.head(20)
    rows=[]
    for _, r in top.iterrows():
        name = (r.get("name") or r.get("Name") or r.get("Ticker")); tick = r.get("Ticker","")
        txt, sc = news_summary(str(name), tick); emoji, dec, why = decision_signal(r, sc, cfg=cfg, context="market")
        if "Acheter" in dec: rows.append({"Name":name,"Ticker":tick,"Var %":round(r[value_col]*100,2),"Décision IA":f"{emoji} {dec}","Raison":why})
    buy_global = pd.DataFrame(rows).head(3)
    urgent_sell = pd.DataFrame()
    if not pf_df.empty:
        prices = fetch_prices(pf_df["Ticker"].tolist(), days=10)
        if not prices.empty:
            last = prices.groupby("Ticker").tail(1)[["Ticker","Close"]].rename(columns={"Close":"Cours"})
            metrics = compute_metrics(prices)
            pf = pf_df.merge(last, on="Ticker", how="left").merge(metrics[["Ticker","MA20","MA50","ATR14"]], on="Ticker", how="left")
            pf["P&L %"] = (pf["Cours"].fillna(0) / pf["PRU"] - 1.0) * 100.0
            decs=[]
            for _, r in pf.iterrows():
                row = {"Close": r.get("Cours", np.nan), "MA20": r.get("MA20", np.nan), "MA50": r.get("MA50", np.nan), "ATR14": r.get("ATR14", np.nan)}
                txt, sc = news_summary(r.get("Nom") or r["Ticker"], r["Ticker"])
                emo, dec, why = decision_signal(row, sc, pnl_pct=r.get("P&L %", np.nan), cfg=cfg, context="portfolio")
                if "VENDRE" in dec: decs.append({"Nom": r.get("Nom") or r["Ticker"], "Ticker": r["Ticker"], "P&L %": round(r["P&L %"],2), "Décision IA": f"{emo} {dec}", "Raison": why})
            urgent_sell = pd.DataFrame(decs).head(3)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("🟢 Achats urgents (global)")
        if buy_global.empty: st.info("Aucune entrée forte détectée.")
        else: st.dataframe(colorize_table(buy_global, "Var %", decision_col="Décision IA"), use_container_width=True)
    with c2:
        st.subheader("🔴 Ventes urgentes (portefeuille)")
        if urgent_sell.empty: st.info("Aucun allègement urgent détecté.")
        else: st.dataframe(colorize_table(urgent_sell, "P&L %", decision_col="Décision IA"), use_container_width=True)
    with c3:
        st.subheader("⚪ À surveiller (global)")
        watch_rows=[]
        for _, r in s.head(20).iterrows():
            name = (r.get("name") or r.get("Name") or r.get("Ticker")); tick=r.get("Ticker","")
            txt, sc = news_summary(str(name), tick); emo, dec, why = decision_signal(r, sc, cfg=cfg, context="market")
            if "Surveiller" in dec: watch_rows.append({"Name":name,"Ticker":tick,"Var %":round(r[value_col]*100,2),"Décision IA":f"{emo} {dec}","Raison":why})
        watch_df = pd.DataFrame(watch_rows).head(5)
        if watch_df.empty: st.info("Rien à surveiller de près pour l’instant.")
        else: st.dataframe(colorize_table(watch_df, "Var %", decision_col="Décision IA"), use_container_width=True)

elif page_selector == "🌍 Marché Global":
    st.title(f"🌍 Marché global — Top {topN} + / − ({periode})")
    all_data = fetch_all_markets(MARKETS_AND_WL, days_hist=days_hist)
    if all_data.empty: st.warning("Aucun univers exploitable."); st.stop()
    valid = all_data.dropna(subset=[value_col]).copy()
    if valid.empty: st.warning("Pas de variations calculables."); st.stop()
    avg = valid[value_col].mean()*100; up=(valid[value_col]>0).sum(); down=(valid[value_col]<0).sum()
    sample_names = valid.sort_values(value_col, ascending=False).head(6).get("name", valid.get("Name", pd.Series(dtype=str))).fillna("").astype(str).tolist()
    sentiments=[news_summary(nm, nm)[1] for nm in sample_names if nm]
    avg_sent = float(np.mean(sentiments)) if sentiments else 0.0
    global_sent = "Le ton du marché est positif." if avg_sent>0.15 else ("Marché sous pression." if avg_sent<-0.15 else "Tonalité neutre.")
    st.markdown(f"<div class='chip'>Résumé global ({periode})</div> <span class='{'pos' if avg>=0 else 'neg'}'>Variation moyenne : {avg:.2f}%</span> — <span class='pos'>{up} hausses</span> / <span class='neg'>{down} baisses</span><br><span class='small'>{global_sent}</span>", unsafe_allow_html=True)
    s = valid.sort_values(value_col, ascending=False); top = s.head(topN); low = s.tail(topN)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏆 Top hausses")
        # bar chart
        d = top.copy(); d["Name"] = d.get("name", d.get("Name", d.get("Ticker", ""))).fillna(d.get("Ticker","")).astype(str); d["pct"] = (d[value_col]*100) if d[value_col].max()<=1 else d[value_col]
        chart = alt.Chart(d).mark_bar().encode(x=alt.X("Name:N", sort="-y"), y=alt.Y("pct:Q"), color=alt.Color("pct:Q", scale=alt.Scale(scheme="greenblue")), tooltip=["Name","Ticker",alt.Tooltip("pct:Q",format=".2f")]).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        df_top = build_analysis_table(top, value_col, cfg, context="market")
        cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
        st.dataframe(colorize_table(df_top[cols], "Variation %", "Sentiment", "Risque CT", "Décision IA"), use_container_width=True)
    with c2:
        st.subheader("📉 Top baisses")
        d = low.copy(); d["Name"] = d.get("name", d.get("Name", d.get("Ticker", ""))).fillna(d.get("Ticker","")).astype(str); d["pct"] = (d[value_col]*100) if d[value_col].max()<=1 else d[value_col]
        chart = alt.Chart(d).mark_bar().encode(x=alt.X("Name:N", sort="-y"), y=alt.Y("pct:Q"), color=alt.Color("pct:Q", scale=alt.Scale(scheme="redpurple")), tooltip=["Name","Ticker",alt.Tooltip("pct:Q",format=".2f")]).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        df_low = build_analysis_table(low, value_col, cfg, context="market")
        cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
        st.dataframe(colorize_table(df_low[cols], "Variation %", "Sentiment", "Risque CT", "Décision IA"), use_container_width=True)
    st.subheader("🎯 Sélections stratégiques (globales) — 3+3")
    hi_glob, lo_glob = pick_candidates(valid, value_col, cfg)
    cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
    st.markdown("**🚀 Haut risque / gains rapides**"); st.dataframe(colorize_table(hi_glob[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)
    st.markdown("**🧱 Moindre risque / gains stables (LT)**"); st.dataframe(colorize_table(lo_glob[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)

elif page_selector == "📊 Détail par Indice":
    st.title(f"📊 Détail — {univers} ({periode})")
    members = build_universe(univers, wl_ls if univers=="LS Exchange" else "")
    if members.empty: st.stop()
    px = fetch_prices(members["ticker"].tolist(), days=days_hist)
    if px.empty: st.stop()
    px = px.merge(members, left_on="Ticker", right_on="ticker", how="left")
    metrics = compute_metrics(px).merge(members, left_on="Ticker", right_on="ticker", how="left")
    valid = metrics.dropna(subset=[value_col]).copy()
    if valid.empty: st.warning("Pas de données suffisantes pour cette période."); st.stop()
    s = valid.sort_values(value_col, ascending=False); top, low = s.head(topN), s.tail(topN)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Hausses")
        d = top.copy(); d["Name"] = d.get("name", d.get("Name", d.get("Ticker", ""))).fillna(d.get("Ticker","")).astype(str); d["pct"] = (d[value_col]*100) if d[value_col].max()<=1 else d[value_col]
        st.altair_chart(alt.Chart(d).mark_bar().encode(x=alt.X("Name:N", sort="-y"), y="pct:Q", color=alt.Color("pct:Q", scale=alt.Scale(scheme="greenblue"))), use_container_width=True)
        df_top = build_analysis_table(top, value_col, cfg, context="market")
        cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
        st.dataframe(colorize_table(df_top[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)
    with c2:
        st.subheader("Baisses")
        d = low.copy(); d["Name"] = d.get("name", d.get("Name", d.get("Ticker", ""))).fillna(d.get("Ticker","")).astype(str); d["pct"] = (d[value_col]*100) if d[value_col].max()<=1 else d[value_col]
        st.altair_chart(alt.Chart(d).mark_bar().encode(x=alt.X("Name:N", sort="-y"), y="pct:Q", color=alt.Color("pct:Q", scale=alt.Scale(scheme="redpurple"))), use_container_width=True)
        df_low = build_analysis_table(low, value_col, cfg, context="market")
        cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
        st.dataframe(colorize_table(df_low[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)
    st.subheader("🎯 Sélections stratégiques — indice (3+3)")
    hi, lo = pick_candidates(valid, value_col, cfg)
    cols = ["Name","Ticker","Cours","Variation %","Sentiment","Risque CT","Conseil","Décision IA","Analyse IA"]
    st.markdown("**🚀 Haut risque / gains rapides**"); st.dataframe(colorize_table(hi[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)
    st.markdown("**🧱 Moindre risque / gains stables (LT)**"); st.dataframe(colorize_table(lo[cols], "Variation %", "Sentiment", "Risque CT","Décision IA"), use_container_width=True)
    st.subheader("📉 Graphique 90 jours avec niveaux")
    all_names = valid.get("name", valid.get("Name", valid.get("Ticker",""))).fillna(valid.get("Ticker","")).astype(str)
    name_to_ticker = dict(zip(all_names, valid["Ticker"]))
    choice = st.selectbox("Sélectionne une action", options=sorted(all_names.unique()))
    if choice:
        row = valid[all_names==choice].iloc[0]; _, sc = news_summary(choice, name_to_ticker[choice])
        ct, mt, lt, buy, stop, tgt, _ = detailed_risk_analysis(row, sc)
        st.caption(f"{ct}  •  {mt}  •  {lt}"); st.caption(f"Plan : Achat~{buy:.2f} / Cible~{tgt:.2f} / Stop~{stop:.2f}")
        data = yf.download(name_to_ticker[choice], period="100d", interval="1d", auto_adjust=False, progress=False).reset_index()
        base = alt.Chart(data).mark_line().encode(x=alt.X("Date:T"), y=alt.Y("Close:Q"), tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=".2f")]).properties(title=f"{choice} — 90 jours")
        rules = alt.Chart(pd.DataFrame({"y":[buy, tgt, stop], "label":["Achat ~","Cible ~","Stop ~"]})).mark_rule().encode(y="y:Q", tooltip=["label:N","y:Q"])
        st.altair_chart(base + rules, use_container_width=True)

else:
    st.title("📁 Mon Portefeuille — P&L, pondérations, IA & décisions (persistant)")
    if "pf_df" not in st.session_state: st.session_state.pf_df = load_portfolio_df()
    st.markdown("#### 🔧 Saisie / modification du portefeuille"); st.caption("Renseigne ticker, quantité et PRU (€). Tu peux éditer directement le tableau.")
    edited = st.data_editor(st.session_state.pf_df, num_rows="dynamic", use_container_width=True,
        column_config={"Ticker": st.column_config.TextColumn(help="Ex: AIR.PA, AAPL, VWCE.DE"),"Quantité": st.column_config.NumberColumn(format="%.4f"),"PRU": st.column_config.NumberColumn(format="%.4f"),"Nom": st.column_config.TextColumn(required=False)})
    st.session_state.pf_df = edited; save_portfolio_df(st.session_state.pf_df)
    colA, colB, colC, colD = st.columns([1,1,1,2])
    with colA:
        refresh = st.button("🔄 Actualiser les cours", use_container_width=True)
    with colB:
        if st.button("🧹 Réinitialiser exemple", use_container_width=True):
            st.session_state.pf_df = load_portfolio_df(); save_portfolio_df(st.session_state.pf_df); st.rerun()
    with colC:
        up = st.file_uploader("📂 Import CSV", type=["csv"])
        if up is not None:
            try:
                imp = pd.read_csv(up); needed = {"Ticker","Quantité","PRU","Nom"}
                if needed.issubset(set(imp.columns)):
                    st.session_state.pf_df = imp[list(needed)]; save_portfolio_df(st.session_state.pf_df); st.success("Portefeuille importé."); st.rerun()
                else: st.error("Colonnes requises : Ticker, Quantité, PRU, Nom")
            except Exception as e: st.error(f"Import invalide: {e}")
    with colD: st.info("Astuce : ajoute une ligne vide pour saisir un nouveau titre.")
    pf = st.session_state.pf_df.copy(); pf = pf[pf["Ticker"].astype(str).str.len()>0]
    if not pf.empty and (refresh or True):
        tickers = pf["Ticker"].dropna().astype(str).tolist()
        prices = fetch_prices(tickers, days=10)
        if prices.empty: st.warning("Impossible de récupérer les cours.")
        else:
            last = prices.groupby("Ticker").tail(1)[["Ticker","Close"]].rename(columns={"Close":"Cours"})
            metrics = compute_metrics(prices)
            pf = pf.merge(last, on="Ticker", how="left").merge(metrics[["Ticker","MA20","MA50","ATR14"]], on="Ticker", how="left")
            pf["Valeur €"] = pf["Quantité"] * pf["Cours"].fillna(0); pf["P&L €"] = (pf["Cours"].fillna(0) - pf["PRU"]) * pf["Quantité"]
            pf["P&L %"] = (pf["Cours"].fillna(0) / pf["PRU"] - 1.0) * 100.0; total_val = pf["Valeur €"].sum()
            pf["Pondération %"] = (pf["Valeur €"] / total_val * 100.0) if total_val>0 else 0.0
            decisions = []
            for _, r in pf.iterrows():
                name = r.get("Nom") or r["Ticker"]
                row = {"Close": r.get("Cours", np.nan), "MA20": r.get("MA20", np.nan), "MA50": r.get("MA50", np.nan), "ATR14": r.get("ATR14", np.nan)}
                txt, sc = news_summary(str(name), str(r["Ticker"]))
                emo, dec, why = decision_signal(row, sc, pnl_pct=r.get("P&L %", np.nan), cfg=load_config(), context="portfolio")
                decisions.append(f"{emo} {dec}")
            pf["Décision IA"] = decisions
            def color_pnl(v):
                if pd.isna(v): return ""
                if v>0: return "background-color:#e8f5e9;color:#0b8f3a"
                if v<0: return "background-color:#ffebee;color:#d5353a"
                return "background-color:#e8f0fe;color:#1e88e5"
            def color_dec(s):
                s=str(s)
                if "🟢" in s: return "background-color:#e8f5e9; color:#0b8f3a"
                if "🔴" in s: return "background-color:#ffebee; color:#d5353a"
                return "background-color:#e8f0fe; color:#1e88e5"
            sty = pf.style.applymap(color_pnl, subset=["P&L €","P&L %"]).applymap(color_dec, subset=["Décision IA"])
            st.markdown("### 📋 Tableau de portefeuille")
            st.dataframe(sty.format({"PRU":"{:.2f}","Cours":"{:.2f}","Valeur €":"{:.2f}","P&L €":"{:.2f}","P&L %":"{:.2f}","Pondération %":"{:.2f}"}), use_container_width=True)
            csv = pf[["Nom","Ticker","Quantité","PRU","Cours","Valeur €","P&L €","P&L %","Pondération %","Décision IA"]].to_csv(index=False).encode("utf-8")
            st.download_button("💾 Exporter le portefeuille (CSV)", data=csv, file_name="portefeuille.csv", mime="text/csv")
            st.markdown("### 🥧 Répartition du portefeuille")
            pie = alt.Chart(pf).mark_arc().encode(theta=alt.Theta("Pondération %:Q"), color=alt.Color("Ticker:N"),
                tooltip=["Ticker","Nom",alt.Tooltip("Pondération %:Q", format=".2f"),alt.Tooltip("Valeur €:Q", format=".2f")]).properties(height=320)
            st.altair_chart(pie, use_container_width=True)
            st.markdown("### 📊 Performance par ligne (P&L €)")
            bars = alt.Chart(pf).mark_bar().encode(x=alt.X("Ticker:N", sort="-y"), y=alt.Y("P&L €:Q"),
                color=alt.Color("P&L €:Q", scale=alt.Scale(scheme="redblue")), tooltip=["Ticker","Nom",alt.Tooltip("P&L €:Q", format=".2f"),alt.Tooltip("P&L %:Q", format=".2f")]).properties(height=320)
            st.altair_chart(bars, use_container_width=True)
            with st.expander("💰 Dividendes (optionnel)", expanded=False):
                cfg_local = load_config(); enable_div = st.toggle("Afficher le module dividendes (expérimental)", value=bool(cfg_local.get("dividendes_enabled", False)))
                if enable_div:
                    cfg_local["dividendes_enabled"] = True; save_config(cfg_local)
                    st.caption("Saisis tes dividendes cumulés par ligne (€/an) pour un premier suivi rapide.")
                    if "Dividendes €/an" not in pf.columns: pf["Dividendes €/an"] = 0.0
                    pf_div = st.data_editor(pf[["Nom","Ticker","Dividendes €/an"]], num_rows="dynamic", use_container_width=True)
                    df_dy = pf.merge(pf_div[["Ticker","Dividendes €/an"]], on="Ticker", how="left")
                    df_dy["Dividend Yield %"] = (df_dy["Dividendes €/an"] / (df_dy["Cours"]*df_dy["Quantité"]).replace(0,np.nan))*100.0
                    st.dataframe(df_dy[["Nom","Ticker","Dividendes €/an","Dividend Yield %"]], use_container_width=True)
                else:
                    cfg_local["dividendes_enabled"] = False; save_config(cfg_local)

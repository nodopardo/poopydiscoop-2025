# app.py
# Poopydiscoop Wrapped — Dashboard interactivo (2024 vs 2025)
# Detecta columnas de días reales, excluye totales/promedios,
# y calcula métricas automáticamente.

import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Poopydiscoop Wrapped", layout="wide")

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())

def _fmt_day_label(col) -> str:
    try:
        dt = pd.to_datetime(col)
        return dt.strftime("%-d-%b")
    except Exception:
        return str(col).strip()

def _is_day_col(colname) -> bool:
    try:
        pd.to_datetime(colname)
        return True
    except Exception:
        pass
    s = str(colname).strip()
    return bool(re.match(r"^\s*\d{1,2}\s*[-/ ]\s*[A-Za-z]{3,}\s*$", s))

@st.cache_data
def load_sheet(xlsx_path: str, sheet_name: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    member_col = None
    for c in df.columns:
        if _norm(c) in {"miembro", "member", "nombre", "participante"}:
            member_col = c
            break
    if member_col is None:
        member_col = df.columns[0]

    df = df.rename(columns={member_col: "Miembro"})
    df["Miembro"] = df["Miembro"].astype(str).str.strip()

    total_mask = df["Miembro"].str.lower().eq("total")
    members = df[~total_mask].copy()

    banned_keywords = {
        "total", "promedio", "average", "kpd", "kgds",
        "cagadasdiarias", "cagadas diarias",
        "totaldecagadas", "total de cagadas",
        "totalkgds", "#kgds",
    }

    day_cols = []
    for c in df.columns:
        if c == "Miembro":
            continue
        nc = _norm(c)
        if any(k in nc for k in banned_keywords):
            continue
        if _is_day_col(c):
            day_cols.append(c)

    try:
        day_cols = sorted(day_cols, key=lambda x: pd.to_datetime(x))
    except Exception:
        pass

    return members, day_cols

st.title("💩 Poopydiscoop Wrapped")
st.caption("Explora el intestino colectivo (2024 vs 2025)")

xlsx_path = "Poopydiscoop.xlsx"
xls = pd.ExcelFile(xlsx_path)
year_sheet = st.selectbox("Selecciona el año", xls.sheet_names, index=len(xls.sheet_names)-1)

members, day_cols = load_sheet(xlsx_path, year_sheet)

all_members = members["Miembro"].tolist()
selected = st.multiselect("Selecciona participantes", all_members, default=all_members)
mf = members[members["Miembro"].isin(selected)].copy()

mf[day_cols] = mf[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

total_kgds = float(mf[day_cols].sum().sum())
avg_per_day = total_kgds / len(day_cols) if day_cols else 0
avg_per_person_day = total_kgds / (len(selected) * len(day_cols)) if selected and day_cols else 0

daily = mf[day_cols].sum(axis=0)
labels = [_fmt_day_label(c) for c in day_cols]
daily.index = labels

c1, c2, c3 = st.columns(3)
c1.metric("KGDs totales", int(total_kgds))
c2.metric("Promedio diario", f"{avg_per_day:.1f}")
c3.metric("Promedio persona/día", f"{avg_per_person_day:.2f}")

fig = px.line(x=daily.index, y=daily.values, labels={"x":"Día","y":"KGDs"})
st.plotly_chart(fig, use_container_width=True)

rank = mf.assign(Total=mf[day_cols].sum(axis=1))[["Miembro","Total"]].sort_values("Total", ascending=False)
st.dataframe(rank, hide_index=True, use_container_width=True)

heat = mf.set_index("Miembro")[day_cols]
heat.columns = labels
fig2 = px.imshow(heat, aspect="auto", labels=dict(x="Día", y="Miembro", color="KGDs"))
st.plotly_chart(fig2, use_container_width=True)

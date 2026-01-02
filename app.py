import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Poopydiscoop Wrapped", layout="wide")

@st.cache_data
def load_sheet(xlsx_path: str, sheet_name: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Normaliza nombres de columnas (por si vienen como fechas, espacios, etc.)
    df.columns = [str(c).strip() for c in df.columns]

    # Detecta la columna de miembros de forma robusta
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().lower())

    member_col = None
    for c in df.columns:
        if norm(c) in {"miembro", "member", "nombre", "participante"}:
            member_col = c
            break
    if member_col is None:
        # fallback: primera columna
        member_col = df.columns[0]

    df = df.rename(columns={member_col: "Miembro"})
    df["Miembro"] = df["Miembro"].astype(str).str.strip()

    # Columnas de totales (acepta varios nombres)
    total_candidates = {"#kgds", "kgds", "total", "total de cagadas", "total kgds"}
    kpd_candidates = {"kpd", "cagadas diarias", "promedio", "promedio diario"}

    tot_col = None
    kpd_col = None
    for c in df.columns:
        if norm(c) in total_candidates:
            tot_col = c
        if norm(c) in kpd_candidates:
            kpd_col = c

    # Columnas por día = todas las que no son Miembro ni totales/promedios
    exclude = {"Miembro"}
    if tot_col: exclude.add(tot_col)
    if kpd_col: exclude.add(kpd_col)
    day_cols = [c for c in df.columns if c not in exclude]

    # Fila total (si existe)
    total_mask = df["Miembro"].str.lower().eq("total")
    total_row = df[total_mask].iloc[0] if total_mask.any() else None

    members = df[~total_mask].copy()

    return members, total_row, day_cols, tot_col, kpd_col


def fmt_day_label(col):
    # Si viene como "2025-12-01 00:00:00", lo pasamos a "1-Dec"
    s = str(col)
    try:
        dt = pd.to_datetime(col)
        return dt.strftime("%-d-%b")  # linux
    except Exception:
        # ya viene como texto
        return s


st.title("💩 Poopydiscoop Wrapped")
st.caption("Explora el intestino colectivo (2024 vs 2025).")

xlsx_path = "Poopydiscoop.xlsx"

# Lista hojas disponibles
xls = pd.ExcelFile(xlsx_path)
year_sheet = st.selectbox("Selecciona el año", xls.sheet_names, index=len(xls.sheet_names)-1)

members, total_row, day_cols, tot_col, kpd_col = load_sheet(xlsx_path, year_sheet)

# Selector de participantes
all_members = members["Miembro"].tolist()
selected = st.multiselect("Selecciona participantes", all_members, default=all_members)
mf = members[members["Miembro"].isin(selected)].copy()

# --------------------------
# KPIs
# --------------------------
colA, colB, colC, colD = st.columns(4)

total_kgds = float(mf[tot_col].sum()) if tot_col else float(mf[day_cols].sum().sum())
avg_per_day = total_kgds / len(day_cols) if len(day_cols) else 0.0
avg_per_person = total_kgds / len(selected) if len(selected) else 0.0

colA.metric("KGDs totales (selección)", f"{int(round(total_kgds))}")
colB.metric("Promedio diario (grupo)", f"{avg_per_day:.1f}")
colC.metric("Promedio por persona", f"{avg_per_person:.1f}")
if total_row is not None:
    # muestra el total global en la hoja
    tot_global = float(total_row[tot_col]) if tot_col else float(members[day_cols].sum().sum())
    colD.metric("KGDs totales (hoja)", f"{int(round(tot_global))}")
else:
    colD.metric("KGDs totales (hoja)", "—")

st.divider()

# --------------------------
# Serie diaria
# --------------------------
st.subheader("📈 Actividad diaria (sumatoria)")
daily = mf[day_cols].sum(axis=0)
daily.index = [fmt_day_label(c) for c in daily.index]

fig = px.line(x=daily.index, y=daily.values, labels={"x": "Día", "y": "KGDs"})
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Ranking
# --------------------------
st.subheader("🏆 Ranking anual")
if tot_col:
    rank = mf[["Miembro", tot_col]].sort_values(tot_col, ascending=False)
    rank = rank.rename(columns={tot_col: "Total"})
else:
    rank = mf.assign(Total=mf[day_cols].sum(axis=1))[["Miembro", "Total"]].sort_values("Total", ascending=False)

st.dataframe(rank, use_container_width=True, hide_index=True)

# --------------------------
# Heatmap
# --------------------------
st.subheader("🟫 Mapa de calor (por día y persona)")
heat = mf.set_index("Miembro")[day_cols]
heat.columns = [fmt_day_label(c) for c in heat.columns]

fig2 = px.imshow(heat, labels=dict(x="Día", y="Miembro", color="KGDs"), aspect="auto")
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# Comparación rápida entre dos personas
# --------------------------
st.subheader("⚔️ Modo rivalidad")
c1, c2 = st.columns(2)
p1 = c1.selectbox("Jugador A", all_members, index=0)
p2 = c2.selectbox("Jugador B", all_members, index=1 if len(all_members) > 1 else 0)

r1 = members[members["Miembro"] == p1].iloc[0]
r2 = members[members["Miembro"] == p2].iloc[0]

s1 = pd.Series(r1[day_cols].values, index=[fmt_day_label(c) for c in day_cols]).astype(float)
s2 = pd.Series(r2[day_cols].values, index=[fmt_day_label(c) for c in day_cols]).astype(float)

df_riv = pd.DataFrame({"Día": s1.index, p1: s1.values, p2: s2.values}).melt(id_vars="Día", var_name="Miembro", value_name="KGDs")
fig3 = px.line(df_riv, x="Día", y="KGDs", color="Miembro")
st.plotly_chart(fig3, use_container_width=True)

st.caption("Tip: si algo se ve raro, revisa que en el repo exista el archivo `Poopydiscoop.xlsx` con las hojas esperadas.")

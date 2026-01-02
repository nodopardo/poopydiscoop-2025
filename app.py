# app.py
# Poopydiscoop Wrapped — Dashboard interactivo (2024 vs 2025)
# - Detecta columnas de días de forma robusta (y excluye "Total de Cagadas", "Cagadas diarias", etc.)
# - Calcula totales y promedios automáticamente (no depende de columnas de totales del Excel)
# - Incluye: KPIs, curva diaria, ranking, heatmap y modo rivalidad

import re
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Poopydiscoop Wrapped", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def _fmt_day_label(col) -> str:
    """Convierte columnas fecha a etiquetas tipo '1-Dec'. Si ya es texto, lo deja."""
    try:
        dt = pd.to_datetime(col)
        # En Linux (Streamlit Cloud) funciona %-d, en Windows no; aquí estamos en Linux.
        return dt.strftime("%-d-%b")
    except Exception:
        return str(col).strip()


def _is_day_col(colname) -> bool:
    """
    Considera 'día' si:
    - el nombre puede parsearse como fecha (Excel suele traer datetimes)
    - o el nombre parece '1-Dec', '31 Dec', '01-Dec', etc.
    """
    # Caso 1: columnas datetime (Excel)
    try:
        pd.to_datetime(colname)
        return True
    except Exception:
        pass

    # Caso 2: texto tipo '1-Dec'
    s = str(colname).strip()
    return bool(re.match(r"^\s*\d{1,2}\s*[-/ ]\s*[A-Za-z]{3,}\s*$", s))


@st.cache_data
def load_sheet(xlsx_path: str, sheet_name: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    # Detectar columna de miembros (robusto)
    member_col = None
    for c in df.columns:
        if _norm(c) in {"miembro", "member", "nombre", "participante"}:
            member_col = c
            break
    if member_col is None:
        member_col = df.columns[0]  # fallback: primera columna

    df = df.rename(columns={member_col: "Miembro"})
    df["Miembro"] = df["Miembro"].astype(str).str.strip()

    # Detectar fila Total (si existe)
    total_mask = df["Miembro"].str.lower().eq("total")
    total_row = df[total_mask].iloc[0] if total_mask.any() else None
    members = df[~total_mask].copy()

    # Excluir columnas que NO son días (aunque no lo parezcan)
    # (esto evita que entren "Total de Cagadas" / "Cagadas diarias" / "KPD" / etc.)
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

    # Ordenar columnas día por fecha si aplica
    try:
        day_cols = sorted(day_cols, key=lambda x: pd.to_datetime(x))
    except Exception:
        # Si no se puede, queda el orden original
        pass

    return members, total_row, day_cols


# -----------------------------
# UI
# -----------------------------
st.title("💩 Poopydiscoop Wrapped")
st.caption("Explora el intestino colectivo (2024 vs 2025). Filtra, compara y revive diciembre.")


# -----------------------------
# Cargar Excel
# -----------------------------
xlsx_path = "Poopydiscoop.xlsx"

try:
    xls = pd.ExcelFile(xlsx_path)
except Exception as e:
    st.error(
        "No pude abrir `Poopydiscoop.xlsx`. "
        "Asegúrate de que el archivo exista en el repo y se llame exactamente así."
    )
    st.stop()

year_sheet = st.selectbox("Selecciona el año (hoja)", xls.sheet_names, index=len(xls.sheet_names) - 1)

members, total_row, day_cols = load_sheet(xlsx_path, year_sheet)

if len(day_cols) == 0:
    st.error(
        "No detecté columnas de días. Revisa que las columnas diarias sean fechas (Excel) "
        "o etiquetas tipo '1-Dec', '2-Dec', ... y que no estén mezcladas con columnas de totales."
    )
    st.stop()

# Selector participantes
all_members = members["Miembro"].tolist()
selected = st.multiselect("Selecciona participantes", all_members, default=all_members)
mf = members[members["Miembro"].isin(selected)].copy()

# Asegurar que días sean numéricos
mf[day_cols] = mf[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# -----------------------------
# KPIs (calculados)
# -----------------------------
st.divider()
colA, colB, colC, colD = st.columns(4)

total_kgds = float(mf[day_cols].sum().sum())
avg_per_day = total_kgds / len(day_cols) if len(day_cols) else 0.0
avg_per_person_per_day = (total_kgds / len(selected) / len(day_cols)) if (len(selected) and len(day_cols)) else 0.0

# Día pico y día mínimo (en la selección)
daily_series = mf[day_cols].sum(axis=0)
day_labels = [_fmt_day_label(c) for c in day_cols]
daily_series.index = day_labels

peak_day = daily_series.idxmax()
peak_val = int(daily_series.max())
min_day = daily_series.idxmin()
min_val = int(daily_series.min())

colA.metric("KGDs totales (selección)", f"{int(round(total_kgds))}")
colB.metric("Promedio diario (grupo)", f"{avg_per_day:.1f}")
colC.metric("Promedio por persona/día", f"{avg_per_person_per_day:.2f}")
colD.metric("Día pico / mínimo", f"{peak_day} ({peak_val}) · {min_day} ({min_val})")

st.divider()


# -----------------------------
# Serie diaria
# -----------------------------
st.subheader("📈 Actividad diaria (sumatoria)")
fig = px.line(
    x=daily_series.index,
    y=daily_series.values,
    labels={"x": "Día", "y": "KGDs"},
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Ranking anual (calculado)
# -----------------------------
st.subheader("🏆 Ranking anual (total mensual)")
rank = mf.assign(Total=mf[day_cols].sum(axis=1))[["Miembro", "Total"]].sort_values("Total", ascending=False)
st.dataframe(rank, use_container_width=True, hide_index=True)


# -----------------------------
# Heatmap
# -----------------------------
st.subheader("🟫 Mapa de calor (por día y persona)")
heat = mf.set_index("Miembro")[day_cols]
heat.columns = day_labels  # etiquetas ya formateadas

fig2 = px.imshow(
    heat,
    labels=dict(x="Día", y="Miembro", color="KGDs"),
    aspect="auto",
)
st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Modo rivalidad
# -----------------------------
st.subheader("⚔️ Modo rivalidad")
c1, c2 = st.columns(2)
p1 = c1.selectbox("Jugador A", all_members, index=0)
p2 = c2.selectbox("Jugador B", all_members, index=1 if len(all_members) > 1 else 0)

r1 = members[members["Miembro"] == p1].copy()
r2 = members[members["Miembro"] == p2].copy()

if len(r1) == 0 or len(r2) == 0:
    st.warning("No pude encontrar uno de los jugadores en esta hoja.")
else:
    r1 = r1.iloc[0]
    r2 = r2.iloc[0]

    s1 = pd.Series(r1[day_cols].values, index=day_labels).astype(float)
    s2 = pd.Series(r2[day_cols].values, index=day_labels).astype(float)

    df_riv = pd.DataFrame({"Día": day_labels, p1: s1.values, p2: s2.values})
    df_riv = df_riv.melt(id_vars="Día", var_name="Miembro", value_name="KGDs")

    fig3 = px.line(df_riv, x="Día", y="KGDs", color="Miembro")
    st.plotly_chart(fig3, use_container_width=True)

    # KPIs rivales
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric(f"Total {p1}", f"{int(s1.sum())}")
    rc2.metric(f"Total {p2}", f"{int(s2.sum())}")
    rc3.metric(f"Pico {p1}", f"{int(s1.max())} ({s1.idxmax()})")
    rc4.metric(f"Pico {p2}", f"{int(s2.max())} ({s2.idxmax()})")


st.caption(
    "Nota: Este dashboard calcula totales y promedios directamente desde las 31 columnas diarias. "
    "Si tu Excel cambia el formato de fechas o encabezados, la detección de días seguirá funcionando."
)

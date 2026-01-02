# app.py
# Poopydiscoop Wrapped — Dashboard interactivo (2024 vs 2025)
# Incluye: KPIs, curva diaria, ranking, heatmap, Rivalidad 1v1,
# "Némesis/Gemelo intestinal" por correlación, y Draft de Equipos (3v3).

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
        return dt.strftime("%-d-%b")  # Streamlit Cloud (Linux) OK
    except Exception:
        return str(col).strip()


def _is_day_col(colname) -> bool:
    """
    Considera 'día' si:
    - el nombre puede parsearse como fecha (Excel suele traer datetimes)
    - o el nombre parece '1-Dec', '31 Dec', etc.
    """
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

    # Columna miembro robusta
    member_col = None
    for c in df.columns:
        if _norm(c) in {"miembro", "member", "nombre", "participante"}:
            member_col = c
            break
    if member_col is None:
        member_col = df.columns[0]  # fallback

    df = df.rename(columns={member_col: "Miembro"})
    df["Miembro"] = df["Miembro"].astype(str).str.strip()

    # Detectar fila Total si existe
    total_mask = df["Miembro"].str.lower().eq("total")
    members = df[~total_mask].copy()

    # Excluir columnas que NO son días (evita que entren "Total de Cagadas", "Cagadas diarias", etc.)
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
        pass

    return members, day_cols


def series_for_person(df_members: pd.DataFrame, day_cols, person: str) -> pd.Series:
    row = df_members[df_members["Miembro"] == person].iloc[0]
    s = pd.Series(row[day_cols].values, index=[_fmt_day_label(c) for c in day_cols]).astype(float)
    return s


def cumulative_race_df(s1: pd.Series, s2: pd.Series, p1: str, p2: str) -> pd.DataFrame:
    c1 = s1.cumsum()
    c2 = s2.cumsum()
    df = pd.DataFrame({"Día": s1.index, p1: c1.values, p2: c2.values})
    return df.melt(id_vars="Día", var_name="Miembro", value_name="Acumulado")


def days_won(s1: pd.Series, s2: pd.Series):
    w1 = int((s1 > s2).sum())
    w2 = int((s2 > s1).sum())
    ties = int((s1 == s2).sum())
    return w1, w2, ties


def biggest_blowout(s1: pd.Series, s2: pd.Series):
    diff = (s1 - s2)
    day = diff.abs().idxmax()
    return day, float(s1[day]), float(s2[day]), float(diff[day])


def corr_top(df_members: pd.DataFrame, day_cols, target: str):
    mat = df_members.set_index("Miembro")[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    # Correlación del patrón diario
    corr = mat.T.corr()
    s = corr[target].drop(index=target).sort_values(ascending=False)
    most_similar = s.head(1)
    most_opposite = s.tail(1)
    return most_similar, most_opposite, s


def weekly_totals(s: pd.Series) -> pd.Series:
    # semanas: 1-7, 8-14, 15-21, 22-28, 29-31
    idx = list(range(1, len(s) + 1))
    buckets = []
    for i, day in enumerate(idx, start=1):
        if day <= 7:
            buckets.append("Semana 1 (1-7)")
        elif day <= 14:
            buckets.append("Semana 2 (8-14)")
        elif day <= 21:
            buckets.append("Semana 3 (15-21)")
        elif day <= 28:
            buckets.append("Semana 4 (22-28)")
        else:
            buckets.append("Final (29-31)")
    df = pd.DataFrame({"bucket": buckets, "val": s.values})
    return df.groupby("bucket")["val"].sum().reindex(
        ["Semana 1 (1-7)", "Semana 2 (8-14)", "Semana 3 (15-21)", "Semana 4 (22-28)", "Final (29-31)"]
    )


# -----------------------------
# UI
# -----------------------------
st.title("💩 Poopydiscoop Wrapped")
st.caption("Explora el intestino colectivo (2024 vs 2025). Filtra, compara y juega con las rivalidades.")

xlsx_path = "Poopydiscoop.xlsx"
xls = pd.ExcelFile(xlsx_path)
year_sheet = st.selectbox("Selecciona el año (hoja)", xls.sheet_names, index=len(xls.sheet_names) - 1)

members, day_cols = load_sheet(xlsx_path, year_sheet)

if len(day_cols) == 0:
    st.error("No detecté columnas de días. Revisa encabezados tipo fechas o '1-Dec'.")
    st.stop()

# Normalizar numéricos
members[day_cols] = members[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

all_members = members["Miembro"].tolist()

tabs = st.tabs(["📊 Resumen", "⚔️ Rivalidad 1 vs 1", "🧠 Némesis / Gemelo", "🎮 Draft (Equipos)"])


# =========================
# TAB 1: RESUMEN
# =========================
with tabs[0]:
    selected = st.multiselect("Selecciona participantes", all_members, default=all_members)
    mf = members[members["Miembro"].isin(selected)].copy()

    total_kgds = float(mf[day_cols].sum().sum())
    avg_per_day = total_kgds / len(day_cols) if len(day_cols) else 0.0
    avg_per_person_day = total_kgds / (len(selected) * len(day_cols)) if selected and day_cols else 0.0

    daily = mf[day_cols].sum(axis=0)
    labels = [_fmt_day_label(c) for c in day_cols]
    daily.index = labels

    peak_day = daily.idxmax()
    peak_val = int(daily.max())
    min_day = daily.idxmin()
    min_val = int(daily.min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KGDs totales (selección)", int(total_kgds))
    c2.metric("Promedio diario (grupo)", f"{avg_per_day:.1f}")
    c3.metric("Promedio persona/día", f"{avg_per_person_day:.2f}")
    c4.metric("Pico / mínimo", f"{peak_day} ({peak_val}) · {min_day} ({min_val})")

    st.subheader("📈 Actividad diaria (sumatoria)")
    fig = px.line(x=daily.index, y=daily.values, labels={"x": "Día", "y": "KGDs"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Ranking anual (total mensual)")
    rank = mf.assign(Total=mf[day_cols].sum(axis=1))[["Miembro", "Total"]].sort_values("Total", ascending=False)
    st.dataframe(rank, hide_index=True, use_container_width=True)

    st.subheader("🟫 Mapa de calor (por día y persona)")
    heat = mf.set_index("Miembro")[day_cols]
    heat.columns = labels
    fig2 = px.imshow(heat, aspect="auto", labels=dict(x="Día", y="Miembro", color="KGDs"))
    st.plotly_chart(fig2, use_container_width=True)


# =========================
# TAB 2: RIVALIDAD 1v1
# =========================
with tabs[1]:
    st.subheader("⚔️ Modo Rivalidad 1 vs 1")
    colA, colB = st.columns(2)
    p1 = colA.selectbox("Jugador A", all_members, index=0, key="p1")
    p2 = colB.selectbox("Jugador B", all_members, index=1 if len(all_members) > 1 else 0, key="p2")

    if p1 == p2:
        st.warning("Elige dos personas distintas para la rivalidad.")
    else:
        s1 = series_for_person(members, day_cols, p1)
        s2 = series_for_person(members, day_cols, p2)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Total {p1}", int(s1.sum()))
        k2.metric(f"Total {p2}", int(s2.sum()))
        w1, w2, ties = days_won(s1, s2)
        k3.metric("Días ganados", f"{p1}: {w1} · {p2}: {w2}")
        k4.metric("Empates", ties)

        day, v1, v2, d = biggest_blowout(s1, s2)
        st.info(f"💥 **Paliza del mes:** {day} — {p1}: {int(v1)} vs {p2}: {int(v2)} (dif: {int(d)})")

        st.markdown("### 📉 Curva diaria (cara a cara)")
        df_line = pd.DataFrame({"Día": s1.index, p1: s1.values, p2: s2.values}).melt(
            id_vars="Día", var_name="Miembro", value_name="KGDs"
        )
        fig = px.line(df_line, x="Día", y="KGDs", color="Miembro")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏁 Carrera acumulada (quién iba ganando día a día)")
        df_race = cumulative_race_df(s1, s2, p1, p2)
        fig2 = px.line(df_race, x="Día", y="Acumulado", color="Miembro")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📦 Totales por semana (para ver el “momentum”)")
        w_s1 = weekly_totals(s1)
        w_s2 = weekly_totals(s2)
        dfw = pd.DataFrame({"Bucket": w_s1.index, p1: w_s1.values, p2: w_s2.values}).melt(
            id_vars="Bucket", var_name="Miembro", value_name="KGDs"
        )
        fig3 = px.bar(dfw, x="Bucket", y="KGDs", color="Miembro", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)


# =========================
# TAB 3: NÉMESIS / GEMELO
# =========================
with tabs[2]:
    st.subheader("🧠 Encuentra tu Gemelo o tu Némesis intestinal")
    target = st.selectbox("Elige participante", all_members, index=0, key="target")

    most_sim, most_opp, corr_series = corr_top(members, day_cols, target)

    sim_name = most_sim.index[0]
    sim_val = float(most_sim.iloc[0])
    opp_name = most_opp.index[0]
    opp_val = float(most_opp.iloc[0])

    c1, c2 = st.columns(2)
    c1.success(f"🤝 **Gemelo intestinal:** {sim_name} (correlación {sim_val:.2f})")
    c2.error(f"😈 **Némesis intestinal:** {opp_name} (correlación {opp_val:.2f})")

    st.markdown("### 🔍 Top 5 más parecidos")
    top5 = corr_series.head(5).reset_index()
    top5.columns = ["Miembro", "Correlación"]
    st.dataframe(top5, hide_index=True, use_container_width=True)

    st.markdown("### 🧩 Comparación visual (target vs gemelo vs némesis)")
    s_t = series_for_person(members, day_cols, target)
    s_g = series_for_person(members, day_cols, sim_name)
    s_n = series_for_person(members, day_cols, opp_name)
    df3 = pd.DataFrame({"Día": s_t.index, target: s_t.values, sim_name: s_g.values, opp_name: s_n.values}).melt(
        id_vars="Día", var_name="Miembro", value_name="KGDs"
    )
    fig = px.line(df3, x="Día", y="KGDs", color="Miembro")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB 4: DRAFT (Equipos)
# =========================
with tabs[3]:
    st.subheader("🎮 Fantasy Poop League — Equipos 3 vs 3")
    st.caption("Arma dos equipos (tríos) cagados. Ideal para apostar un cacique")

    left, right = st.columns(2)
    team_a = left.multiselect("Equipo A (elige 3)", all_members, default=all_members[:3], key="team_a")
    team_b = right.multiselect("Equipo B (elige 3)", all_members, default=all_members[3:6], key="team_b")

    if len(team_a) != 3 or len(team_b) != 3:
        st.warning("Para que sea Draft real, elige exactamente 3 en cada equipo.")
    else:
        dfA = members[members["Miembro"].isin(team_a)][day_cols].sum(axis=0)
        dfB = members[members["Miembro"].isin(team_b)][day_cols].sum(axis=0)

        sA = pd.Series(dfA.values, index=[_fmt_day_label(c) for c in day_cols]).astype(float)
        sB = pd.Series(dfB.values, index=[_fmt_day_label(c) for c in day_cols]).astype(float)

        totalA = int(sA.sum())
        totalB = int(sB.sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Equipo A", totalA)
        k2.metric("Total Equipo B", totalB)
        k3.metric("Ganador", "Equipo A" if totalA > totalB else ("Equipo B" if totalB > totalA else "Empate"))

        wA, wB, ties = days_won(sA, sB)
        st.info(f"📆 **Días ganados:** Equipo A {wA} · Equipo B {wB} · Empates {ties}")

        st.markdown("### 📈 Curva diaria (Equipos)")
        df_line = pd.DataFrame({"Día": sA.index, "Equipo A": sA.values, "Equipo B": sB.values}).melt(
            id_vars="Día", var_name="Equipo", value_name="KGDs"
        )
        fig = px.line(df_line, x="Día", y="KGDs", color="Equipo")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏁 Carrera acumulada (Equipos)")
        df_race = cumulative_race_df(sA, sB, "Equipo A", "Equipo B")
        fig2 = px.line(df_race, x="Día", y="Acumulado", color="Miembro")
        st.plotly_chart(fig2, use_container_width=True)


st.caption(
    "Nota: El dashboard detecta automáticamente las 31 columnas diarias y calcula totales/promedios desde ahí. "
    "No usa 'Total de Cagadas' ni 'Cagadas diarias' para evitar distorsiones."
)

# app.py
# Poopydiscoop Wrapped — Dashboard interactivo (2024 vs 2025)
# Incluye:
# - Resumen: KPIs, curva diaria, ranking, heatmap + interpretación automática
# - Rivalidad 1v1
# - Némesis/Gemelo
# - Draft 3v3
# - Fantasy Poop League
# - 🏅 Premiación: SOLO mensajes personalizados + ficha con stats

import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Poopydiscoop Wrapped", layout="wide")

APP_VERSION = "2026-01-02-v4-custom-only-fixed"


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def _fmt_day_label(col) -> str:
    try:
        dt = pd.to_datetime(col)
        return dt.strftime("%-d-%b")  # Linux OK
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
def load_sheet(xlsx_path: str, sheet_name: str, version: str):
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
    if "Miembro" not in df.columns:
        df.insert(0, "Miembro", df.iloc[:, 0].astype(str))

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

    members[day_cols] = members[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return members, day_cols


def series_for_person(df_members: pd.DataFrame, day_cols, person: str) -> pd.Series:
    row = df_members[df_members["Miembro"] == person].iloc[0]
    return pd.Series(row[day_cols].values, index=[_fmt_day_label(c) for c in day_cols]).astype(float)


def cumulative_race_df(s1: pd.Series, s2: pd.Series, p1: str, p2: str) -> pd.DataFrame:
    df = pd.DataFrame({"Día": s1.index, p1: s1.cumsum().values, p2: s2.cumsum().values})
    return df.melt(id_vars="Día", var_name="Miembro", value_name="Acumulado")


def days_won(s1: pd.Series, s2: pd.Series):
    return int((s1 > s2).sum()), int((s2 > s1).sum()), int((s1 == s2).sum())


def biggest_blowout(s1: pd.Series, s2: pd.Series):
    diff = (s1 - s2)
    day = diff.abs().idxmax()
    return day, float(s1[day]), float(s2[day]), float(diff[day])


def corr_top(df_members: pd.DataFrame, day_cols, target: str):
    mat = df_members.set_index("Miembro")[day_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    corr = mat.T.corr()
    s = corr[target].drop(index=target).sort_values(ascending=False)
    return s.head(1), s.tail(1), s


def weekly_totals(s: pd.Series) -> pd.Series:
    idx = list(range(1, len(s) + 1))
    buckets = []
    for day in idx:
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


def totals_by_period(df_members: pd.DataFrame, day_cols, roster: list[str]) -> pd.Series:
    if not roster:
        return pd.Series([0, 0, 0, 0, 0], index=[
            "Semana 1 (1-7)", "Semana 2 (8-14)", "Semana 3 (15-21)", "Semana 4 (22-28)", "Final (29-31)"
        ])
    team_daily = df_members[df_members["Miembro"].isin(roster)][day_cols].sum(axis=0)
    s = pd.Series(team_daily.values, index=[_fmt_day_label(c) for c in day_cols]).astype(float)
    return weekly_totals(s)


def make_schedule(managers: list[str], periods: list[str]) -> pd.DataFrame:
    rows = []
    for p in periods:
        ms = managers[:]
        pairs = list(zip(ms[0::2], ms[1::2]))
        used = set()
        for a, b in pairs:
            rows.append({"Periodo": p, "Local": a, "Visitante": b})
            used.add(a); used.add(b)
        if len(ms) % 2 == 1:
            bye = [m for m in ms if m not in used][0]
            rows.append({"Periodo": p, "Local": bye, "Visitante": "BYE"})
    return pd.DataFrame(rows)


def member_stats(df_members: pd.DataFrame, day_cols) -> pd.DataFrame:
    totals = df_members[day_cols].sum(axis=1)
    kpd = totals / len(day_cols)
    active = (df_members[day_cols] > 0).sum(axis=1)
    zeros = len(day_cols) - active
    peak_val = df_members[day_cols].max(axis=1)

    peak_day = []
    for _, row in df_members.iterrows():
        vals = row[day_cols].values.astype(float)
        i = int(np.argmax(vals))
        peak_day.append(_fmt_day_label(day_cols[i]))

    return pd.DataFrame({
        "Miembro": df_members["Miembro"].values,
        "Total": totals.round(0).astype(int),
        "KPD": kpd.round(2),
        "Días activos": active.astype(int),
        "Días 0": zeros.astype(int),
        "Pico": peak_val.round(0).astype(int),
        "Pico (día)": peak_day,
    })


def interpret_summary(daily: pd.Series, rank_df: pd.DataFrame) -> list[str]:
    lines = []
    peak_day = daily.idxmax()
    peak_val = int(daily.max())
    min_day = daily.idxmin()
    min_val = int(daily.min())
    top1 = rank_df.iloc[0]
    top2 = rank_df.iloc[1] if len(rank_df) > 1 else None

    lines.append(f"📈 **Curva diaria:** pico el **{peak_day}** con **{peak_val} KGDs**.")
    lines.append(f"🧊 **Día más tranqui:** **{min_day}** con **{min_val} KGDs**.")
    lines.append(f"🏆 **Ranking:** líder **{top1['Miembro']}** con **{int(top1['Total'])} KGDs**.")
    if top2 is not None:
        lines.append(f"🥈 Segundo: **{top2['Miembro']}** con **{int(top2['Total'])} KGDs**.")
    lines.append("🟫 **Heatmap:** más oscuro = más actividad; más claro = días en cero.")
    return lines


# -----------------------------
# Mensajes personalizados (Premiación) — SOLO ESTOS
# emoji + premio + mensaje
# -----------------------------
CUSTOM_AWARDS = {
    "Nico": ("👑", "El desafiante del trono", "2025 fue su toma de poder con sed de historia y electrolit, un reinado gracias a la gastroenteritis."),
    "Fredo": ("🏛️", "El rey emérito", "La corona puede rotar, pero la leyenda queda. Todos sabemos que los Warriors merecían la final del 2016."),
    "Andy": ("🥉", "La medalla escatológica", "Entrando al top 3 con estilo. Un ano de bronce sin hacer escándalo."),
    "Miguel": ("🧠", "El regulador filosófico", "Nunca extremo, siempre presente. Perdiendo podio, pero ganando peso. Glow-down: -13 KGDs este año."),
    "Didi": ("📺", "La racha intestilente", "Pausas calculadas, regresos con estilo. La mejor serie del 2025."),
    "Marcos Daniel": ("💔", "La novia tóxica", "Picos inesperados y siempre vuelve."),
    "Luis": ("📈", "El maestro del glow-up", "Sí me entiende. Glow-up +14 KGDs este año."),
    "Vagner": ("🛞", "El nuevo motor wotor", "Entró con buen promedio y cambió el tablero."),
    "Carlos": ("🔥", "El resurgido", "2025 fue el ano del regreso. Un resultado a la altura que merece. Glow-up +15 KGDs este año."),
    "Simón": ("⚖️", "El equilibrio rectal", "Ni caos ni rigidez, solo flujo (anal). Su gemelo intestinal es Andy."),
    "Misa": ("📉", "El metódico descendente", "Regularidad tranquila, sin picos innecesarios. Glow-down: -12 KGDs este año."),
    "Esteban": ("🥷", "El ninja silencioso", "Aparece sin avisar y suma sin drama. Plot twist: -15 KGDs en comparación al año pasado."),
    "Jorge": ("🧘", "El ingeniero Zen", "Optimiza energía, minimiza ruido, maximiza calma, estriñe los fines de año."),
    "Marcos Javier": ("🤫", "El sigiloso", "Menos es más, siempre. Misa es su gemelo intestinal."),
    "Pablo": ("👻", "El fantasma de las navidades presentes", "Poco frecuente, pero inolvidable que cague 2 veces el 3 y 10 de diciembre."),
    "Tama": ("🎬", "Stan Lee en Marvel", "Cameo suave: dejaste dato. Y cómo es que tu máxima cagada diaria es de 1."),
}


# -----------------------------
# UI
# -----------------------------
st.title("💩 Poopydiscoop Wrapped")
st.caption("Explora el intestino colectivo durante diciembre (2024 y 2025). Filtra, compara y juega con las rivalidades.")

xlsx_path = "Poopydiscoop.xlsx"
xls = pd.ExcelFile(xlsx_path)
year_sheet = st.selectbox("Selecciona el año (hoja)", xls.sheet_names, index=len(xls.sheet_names) - 1)

members, day_cols = load_sheet(xlsx_path, year_sheet, APP_VERSION)

if len(day_cols) == 0:
    st.error("No detecté columnas de días. Revisa encabezados tipo fechas o '1-Dec'.")
    st.stop()

all_members = members["Miembro"].tolist()

tabs = st.tabs([
    "📊 Resumen",
    "🏅 Premiación",
    "⚔️ Rivalidad 1 vs 1",
    "🧠 Némesis / Gemelo",
    "🎮 Draft (3 vs 3)",
    "🏈 Fantasy Poop League"
])


# =========================
# TAB 1: RESUMEN
# =========================
with tabs[0]:
    selected = st.multiselect("Selecciona participantes", all_members, default=all_members)
    mf = members[members["Miembro"].isin(selected)].copy()

    total_kgds = float(mf[day_cols].sum().sum())
    avg_per_day = total_kgds / len(day_cols)
    avg_per_person_day = total_kgds / (len(selected) * len(day_cols)) if selected else 0.0

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

    with st.expander("🗣️ Interpretación dummy de los gráficos", expanded=True):
        for line in interpret_summary(daily, rank):
            st.markdown(f"- {line}")


# =========================
# TAB 2: PREMIACIÓN (SOLO PERSONALIZADO)
# =========================
with tabs[1]:
    st.subheader("🏅 Premiación individual (solo canon oficial)")
    st.caption("Aquí no hay IA inventando: solo tus premios + tus textos, con datos reales al lado.")

    stats = member_stats(members, day_cols)

    # Tabla completa + columna Premio (si existe)
    rows = []
    for _, row in stats.iterrows():
        name = row["Miembro"]
        if name in CUSTOM_AWARDS:
            emoji, title, _msg = CUSTOM_AWARDS[name]
            premio = f"{emoji} {title}"
        else:
            premio = "—"
        rows.append({
            "Miembro": name,
            "Premio": premio,
            "Total": int(row["Total"]),
            "KPD": float(row["KPD"]),
            "Días activos": int(row["Días activos"]),
            "Días 0": int(row["Días 0"]),
            "Pico (día)": f"{int(row['Pico'])} ({row['Pico (día)']})"
        })

    awards_table = pd.DataFrame(rows).sort_values(["Total"], ascending=False)

    st.markdown("### 📋 Tabla de premiación")
    st.dataframe(awards_table, hide_index=True, use_container_width=True)

    st.markdown("### 🧾 Ficha individual")
    chosen = st.selectbox("Elige participante", awards_table["Miembro"].tolist(), index=0)

    r = stats.set_index("Miembro").loc[chosen]

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Total", int(r["Total"]))
    a2.metric("KPD", float(r["KPD"]))
    a3.metric("Días activos", int(r["Días activos"]))
    a4.metric("Días 0", int(r["Días 0"]))
    a5.metric("Pico (día)", f"{int(r['Pico'])} ({r['Pico (día)']})")

    if chosen in CUSTOM_AWARDS:
        emoji, title, msg = CUSTOM_AWARDS[chosen]
        st.success(f"**{emoji} {title}**")
        st.info(msg)
    else:
        st.warning("Este participante aún no tiene mensaje personalizado en `CUSTOM_AWARDS`.")

    # Mini gráfico individual
    s = series_for_person(members, day_cols, chosen)
    df_line = pd.DataFrame({"Día": s.index, "KGDs": s.values})
    fig = px.line(df_line, x="Día", y="KGDs", markers=True)
    st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB 3: RIVALIDAD 1v1
# =========================
with tabs[2]:
    st.subheader("⚔️ Modo Rivalidad 1 vs 1")
    colA, colB = st.columns(2)
    p1 = colA.selectbox("KGDor A", all_members, index=0, key="p1")
    p2 = colB.selectbox("KGDor B", all_members, index=1 if len(all_members) > 1 else 0, key="p2")

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

        st.markdown("### 🏁 Carrera acumulada")
        df_race = cumulative_race_df(s1, s2, p1, p2)
        fig2 = px.line(df_race, x="Día", y="Acumulado", color="Miembro")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📦 Totales por semana (momentum)")
        w_s1 = weekly_totals(s1)
        w_s2 = weekly_totals(s2)
        dfw = pd.DataFrame({"Bucket": w_s1.index, p1: w_s1.values, p2: w_s2.values}).melt(
            id_vars="Bucket", var_name="Miembro", value_name="KGDs"
        )
        fig3 = px.bar(dfw, x="Bucket", y="KGDs", color="Miembro", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)


# =========================
# TAB 4: NÉMESIS / GEMELO
# =========================
with tabs[3]:
    st.subheader("🧠 Encuentra tu gemelo o tu némesis intestinal")
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

    st.markdown("### 🧩 Comparación visual")
    s_t = series_for_person(members, day_cols, target)
    s_g = series_for_person(members, day_cols, sim_name)
    s_n = series_for_person(members, day_cols, opp_name)
    df3 = pd.DataFrame({"Día": s_t.index, target: s_t.values, sim_name: s_g.values, opp_name: s_n.values}).melt(
        id_vars="Día", var_name="Miembro", value_name="KGDs"
    )
    fig = px.line(df3, x="Día", y="KGDs", color="Miembro")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB 5: DRAFT 3v3
# =========================
with tabs[4]:
    st.subheader("🎮 Draft Mode — Equipos 3 vs 3")
    left, right = st.columns(2)
    team_a = left.multiselect("Equipo A (elige 3)", all_members, default=all_members[:3], key="team_a")
    team_b = right.multiselect("Equipo B (elige 3)", all_members, default=all_members[3:6], key="team_b")

    if len(team_a) != 3 or len(team_b) != 3:
        st.warning("Elige exactamente 3 en cada equipo.")
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

        df_line = pd.DataFrame({"Día": sA.index, "Equipo A": sA.values, "Equipo B": sB.values}).melt(
            id_vars="Día", var_name="Equipo", value_name="KGDs"
        )
        fig = px.line(df_line, x="Día", y="KGDs", color="Equipo")
        st.plotly_chart(fig, use_container_width=True)

        df_race = cumulative_race_df(sA, sB, "Equipo A", "Equipo B")
        fig2 = px.line(df_race, x="Día", y="Acumulado", color="Miembro")
        st.plotly_chart(fig2, use_container_width=True)


# =========================
# TAB 6: FANTASY
# =========================
with tabs[5]:
    st.subheader("🏈 Fantasy Poop League (para apostar)")
    st.caption("Cada manager draftea un roster. 5 jornadas: 4 semanas + final. Tabla por puntos + H2H.")

    n = st.slider("Número de managers", min_value=2, max_value=6, value=4, step=1)
    roster_size = st.slider("Tamaño del roster por manager", min_value=2, max_value=4, value=3, step=1)

    mgr_cols = st.columns(n)
    managers = []
    for i in range(n):
        managers.append(mgr_cols[i].text_input(f"Manager {i+1}", value=f"Manager {i+1}", key=f"mgr_{i}").strip() or f"Manager {i+1}")

    st.markdown("### 🧩 Draft del roster")
    rosters = {}
    used = []
    for i, m in enumerate(managers):
        with st.expander(f"📌 {m} — elige {roster_size}", expanded=(i == 0)):
            picks = st.multiselect(
                f"Roster de {m}",
                options=all_members,
                default=[],
                key=f"roster_{i}"
            )
            rosters[m] = picks
            used.extend(picks)

    dupes = sorted({x for x in used if used.count(x) > 1})
    if dupes:
        st.error(f"🚫 Jugadores repetidos en la liga: {', '.join(dupes)}. Ajusten rosters para que sean únicos.")

    periods = ["Semana 1 (1-7)", "Semana 2 (8-14)", "Semana 3 (15-21)", "Semana 4 (22-28)", "Final (29-31)"]

    points = {}
    for m, roster in rosters.items():
        points[m] = totals_by_period(members, day_cols, roster)

    df_points = pd.DataFrame(points).T
    df_points["Total"] = df_points.sum(axis=1)

    st.markdown("### 🧾 Tabla Fantasy (puntos por periodo)")
    st.dataframe(df_points.sort_values("Total", ascending=False), use_container_width=True)

    st.markdown("### 🏆 Leaderboard (Total)")
    df_lb = df_points["Total"].sort_values(ascending=False).reset_index()
    df_lb.columns = ["Manager", "Total"]
    fig_lb = px.bar(df_lb, x="Manager", y="Total")
    st.plotly_chart(fig_lb, use_container_width=True)

    st.markdown("### 📅 Jornadas (Head-to-Head)")
    schedule = make_schedule(managers, periods)

    rows = []
    for _, r in schedule.iterrows():
        period = r["Periodo"]
        a = r["Local"]
        b = r["Visitante"]
        if b == "BYE":
            rows.append({"Periodo": period, "Local": a, "Visitante": "BYE", "Puntos Local": df_points.loc[a, period], "Puntos Visitante": 0, "Ganador": a})
        else:
            pa = df_points.loc[a, period]
            pb = df_points.loc[b, period]
            win = a if pa > pb else (b if pb > pa else "Empate")
            rows.append({"Periodo": period, "Local": a, "Visitante": b, "Puntos Local": pa, "Puntos Visitante": pb, "Ganador": win})

    df_h2h = pd.DataFrame(rows)
    st.dataframe(df_h2h, use_container_width=True, hide_index=True)

    st.markdown("### ✅ Tabla de victorias (H2H)")
    wins = {m: 0 for m in managers}
    ties = {m: 0 for m in managers}
    for _, r in df_h2h.iterrows():
        if r["Visitante"] == "BYE":
            wins[r["Ganador"]] += 1
        elif r["Ganador"] == "Empate":
            ties[r["Local"]] += 1
            ties[r["Visitante"]] += 1
        else:
            wins[r["Ganador"]] += 1

    df_w = pd.DataFrame({
        "Manager": managers,
        "Victorias": [wins[m] for m in managers],
        "Empates": [ties[m] for m in managers],
        "Total Fantasy": [df_points.loc[m, "Total"] for m in managers]
    }).sort_values(["Victorias", "Total Fantasy"], ascending=False)

    st.dataframe(df_w, use_container_width=True, hide_index=True)

    st.markdown("### 💸 Ideas de apuesta")
    st.write("- Último en H2H invita a bigarepa solo a las 4 a.m.")
    st.write("- Campeón global elige lote en la pastora para acampar.")
    st.write("- Empate global: duelo 1v1 entre los dos mejores managers (usa el tab Rivalidad).")


st.caption(
    "Nota: El dashboard detecta columnas diarias y calcula totales/promedios desde ahí. "
    "No usa columnas tipo 'Total de Cagadas' ni 'Cagadas diarias'."
)

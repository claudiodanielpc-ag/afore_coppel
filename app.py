import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from names_dataset import NameDataset, NameWrapper
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Leads Afore Coppel",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
    .stSelectbox, .stSlider, .stButton, h1, h2, h3, p, div {
        font-family: 'Century Gothic', 'Nunito', 'Trebuchet MS', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.image("logo_ag_transp.png", width=180)
with col_titulo:
    st.title("Resultados preliminares de leads de contacto — Afore Coppel")

resumen_placeholder = st.empty()


# ── Helpers ─────────────────────────────────────────────────────────

@st.cache_resource
def get_name_dataset():
    return NameDataset()


def inferir_sexo(nombre_completo, nd):
    partes = str(nombre_completo).strip().split()
    for nombre in partes[:2]:
        w = NameWrapper(nd.search(nombre))
        if w.gender in ('Male', 'Female'):
            return 'Hombre' if w.gender == 'Male' else 'Mujer'
    return 'Indeterminado'


@st.cache_resource
def get_engine():
    # Escribe el .pem en un archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as pem_file:
        pem_file.write(st.secrets["SSH_KEY_CONTENT"])
        pem_path = pem_file.name
    os.chmod(pem_path, 0o600)

    # Escribe un .env temporal con las credenciales
    env_content = "\n".join([
        f'SSH_HOST={st.secrets["SSH_HOST"]}',
        f'SSH_PORT={st.secrets["SSH_PORT"]}',
        f'SSH_USER={st.secrets["SSH_USER"]}',
        f'SSH_KEY_FILE={pem_path}',
        f'SSH_KEY_PASSPHRASE={st.secrets["SSH_KEY_PASSPHRASE"]}',
        f'REMOTE_DB_HOST={st.secrets["REMOTE_DB_HOST"]}',
        f'REMOTE_DB_PORT={st.secrets["REMOTE_DB_PORT"]}',
        f'DB_USER={st.secrets["DB_USER"]}',
        f'DB_PASS={st.secrets["DB_PASS"]}',
    ])
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as env_file:
        env_file.write(env_content)
        env_path = env_file.name

    from cd_base import ConexionBD
    bd = ConexionBD(env_path)
    return bd.conectar("landing_ucc")


@st.cache_data(ttl=900)  # refresca cada 15 minutos
def load_data():
    engine = get_engine()
    nd = get_name_dataset()
    df = pd.read_sql(
        "SELECT * FROM landing_interesados_afore WHERE created_at >= '2026-04-15 10:00:00'",
        engine,
    )
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['correo'] = df['correo'].str.strip()
    df = df.drop_duplicates(subset='correo', keep='first').reset_index(drop=True)
    df['sexo'] = df['nombre'].apply(lambda x: inferir_sexo(x, nd))
    return df


# ── Carga de datos ───────────────────────────────────────────────────

with st.spinner("Conectando a la base de datos..."):
    df = load_data()

if st.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Filtro de fechas ─────────────────────────────────────────────────

fecha_min = df['created_at'].dt.date.min()
fecha_max = df['created_at'].dt.date.max()

col_f1, col_f2, _ = st.columns([1, 1, 3])
with col_f1:
    fecha_inicio = st.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
with col_f2:
    fecha_fin = st.date_input("Hasta", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

df_f = df[
    (df['created_at'].dt.date >= fecha_inicio) &
    (df['created_at'].dt.date <= fecha_fin)
]

# ── Métricas ─────────────────────────────────────────────────────────

total = len(df_f)
inscripcion = (df_f['interesado_en'] == 'Iniciar Proceso de inscripción').sum()
info = (df_f['interesado_en'] == 'Solicitar Información').sum()
hombres = (df_f['sexo'] == 'Hombre').sum()
mujeres = (df_f['sexo'] == 'Mujer').sum()

@st.cache_data(ttl=900)
def generar_resumen_ia(total, inscripcion, info, hombres, mujeres, fecha_ini, fecha_fin):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    pct = lambda n: f"{n/total*100:.1f}%" if total else "N/D"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": (
            f"Eres un analista de datos. Escribe un resumen ejecutivo breve (2-3 oraciones) "
            f"en español sobre los leads de Afore Coppel del {fecha_ini} al {fecha_fin}:\n"
            f"- Total leads: {total}\n"
            f"- Interesados en inscripción: {inscripcion} ({pct(inscripcion)})\n"
            f"- Solicitan información: {info} ({pct(info)})\n"
            f"- Hombres: {hombres} ({pct(hombres)}), Mujeres: {mujeres} ({pct(mujeres)})\n"
            "Sé directo y profesional. No uses listas, solo prosa."
        )}]
    )
    return msg.content[0].text

with resumen_placeholder.container():
    with st.spinner("Generando resumen..."):
        resumen = generar_resumen_ia(total, inscripcion, info, hombres, mujeres, fecha_inicio, fecha_fin)
    st.info(resumen)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total leads", total)
c2.metric("Proceso de inscripción", inscripcion, f"{inscripcion/total*100:.1f}%" if total else "—")
c3.metric("Solicitar información", info, f"{info/total*100:.1f}%" if total else "—")
c4.metric("Hombres", hombres, f"{hombres/total*100:.1f}%" if total else "—")
c5.metric("Mujeres", mujeres, f"{mujeres/total*100:.1f}%" if total else "—")

st.divider()

# ── Leads por hora ────────────────────────────────────────────────────

st.subheader("Leads por hora")

tipo = st.selectbox(
    "Tipo de interés",
    ["Todos", "Iniciar Proceso de inscripción", "Solicitar Información"],
)

df_hora = df_f if tipo == "Todos" else df_f[df_f['interesado_en'] == tipo]

por_hora = (
    df_hora
    .groupby(df_hora['created_at'].dt.floor('h'))
    .size()
    .reset_index(name='cantidad')
)

fig_hora = go.Figure(go.Scatter(
    x=por_hora['created_at'],
    y=por_hora['cantidad'],
    mode='lines+markers',
    line=dict(color='steelblue', width=2),
    marker=dict(size=6),
))
fig_hora.update_layout(
    xaxis_title="Hora",
    yaxis_title="Leads",
    height=320,
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_hora, use_container_width=True)

st.divider()

# ── Programas más solicitados ─────────────────────────────────────────

st.subheader("Programas más solicitados")

col_t1, col_t2 = st.columns(2)

for col, label in [
    (col_t1, "Iniciar Proceso de inscripción"),
    (col_t2, "Solicitar Información"),
]:
    sub = df_f[df_f['interesado_en'] == label]
    tabla = sub['programa_interes'].value_counts().reset_index()
    tabla.columns = ['Programa', 'Cantidad']
    tabla['%'] = (tabla['Cantidad'] / tabla['Cantidad'].sum() * 100).round(1).astype(str) + '%'
    with col:
        st.caption(label)
        st.dataframe(tabla, hide_index=True, use_container_width=True)

st.divider()

# ── Proyección ARIMA ──────────────────────────────────────────────────

st.subheader("Correos acumulados manifestando interés en inscribirse — ARIMA(1,1,0)")

df_arima = df[df['interesado_en'] == 'Iniciar Proceso de inscripción'].copy()
por_hora_ac = (
    df_arima
    .set_index('created_at')
    .resample('h')
    .size()
    .rename('cantidad')
    .to_frame()
)
por_hora_ac['acumulado'] = por_hora_ac['cantidad'].cumsum()

modelo = ARIMA(por_hora_ac['acumulado'], order=(1, 1, 0))
resultado = modelo.fit()
forecast = resultado.get_forecast(steps=48)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)

total_obs = por_hora_ac['acumulado'].iloc[-1]
total_est = pred_mean.iloc[-1]
incremento = total_est - total_obs

ca1, ca2, ca3 = st.columns(3)
ca1.metric("Acumulado actual", f"{total_obs:.0f}")
ca2.metric("Estimado +48h", f"{total_est:.0f}")
ca3.metric("Incremento esperado", f"+{incremento:.0f}")

fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(
    x=por_hora_ac.index,
    y=por_hora_ac['acumulado'],
    mode='lines+markers',
    name='Observado',
    line=dict(color='steelblue', width=2),
))
fig_arima.add_trace(go.Scatter(
    x=pred_mean.index,
    y=pred_mean,
    mode='lines+markers',
    name='Proyección ARIMA',
    line=dict(color='tomato', width=2, dash='dash'),
))
fig_arima.add_trace(go.Scatter(
    x=list(pred_ci.index) + list(pred_ci.index[::-1]),
    y=list(pred_ci.iloc[:, 1]) + list(pred_ci.iloc[:, 0][::-1]),
    fill='toself',
    fillcolor='rgba(255,99,71,0.15)',
    line=dict(color='rgba(0,0,0,0)'),
    name='IC 95%',
    showlegend=True,
))
fig_arima.update_layout(
    xaxis_title="Hora",
    yaxis_title="Correos acumulados",
    height=420,
    margin=dict(t=20, b=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
)
st.plotly_chart(fig_arima, use_container_width=True)

st.caption(
    f"Modelo ARIMA(1,1,0) · Datos desde {fecha_min} · Caché: 15 min  \n"
    "**Nota:** Los correos contabilizados reflejan intención de inscripción manifestada por el usuario, "
    "no inscripciones efectivas confirmadas. El número real de alumnos inscritos puede diferir."
)

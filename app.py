import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from names_dataset import NameDataset, NameWrapper
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Interesados Afore Coppel",
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
    st.markdown(
        "<h1 style='line-height:1.2'>Resultados preliminares de interesados en oferta académica<br>"
        "Academia Global — Afore Coppel</h1>",
        unsafe_allow_html=True,
    )

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
    df = df[~df['nombre'].isin(['1', 'prdf'])].reset_index(drop=True)
    return df


# ── Carga de datos ───────────────────────────────────────────────────

with st.spinner("Conectando a la base de datos..."):
    df = load_data()

if st.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

st.divider()

_MESES = ['enero','febrero','marzo','abril','mayo','junio',
          'julio','agosto','septiembre','octubre','noviembre','diciembre']

def fmt_fecha(d):
    return f"{d.day} de {_MESES[d.month - 1]} de {d.year}"

# ── Filtro de fechas ─────────────────────────────────────────────────

from datetime import date as _date

fecha_min = df['created_at'].dt.date.min()
fecha_max = df['created_at'].dt.date.max()
hoy = _date.today()

col_f1, col_f2, _ = st.columns([1, 1, 3])
with col_f1:
    fecha_inicio = st.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=hoy)
with col_f2:
    fecha_fin = st.date_input("Hasta", value=hoy, min_value=fecha_min, max_value=hoy)

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
indeterminado = (df_f['sexo'] == 'Indeterminado').sum()

@st.cache_data(ttl=900)
def generar_resumen_ia(total, inscripcion, info, hombres, mujeres,
                       fecha_ini, fecha_fin, total_obs, total_est, incremento, pen_actual, tasa_requerida):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    pct = lambda n: f"{n/total*100:.1f}%" if total else "N/D"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": (
            f"Eres un analista de datos. Escribe un resumen ejecutivo breve (2-3 oraciones) "
            f"en español sobre los interesados de Afore Coppel del {fecha_ini} al {fecha_fin}. "
            f"Todos los registros son únicos (se omitieron duplicados por correo electrónico):\n"
            f"- Total registros únicos: {total}\n"
            f"- Interesados en inscripción: {inscripcion} ({pct(inscripcion)})\n"
            f"- Solicitan información: {info} ({pct(info)})\n"
            f"- Hombres: {hombres} ({pct(hombres)}), Mujeres: {mujeres} ({pct(mujeres)})\n"
            f"- Tasa de contacto única sobre población potencial: {pen_actual:.3f}%\n"
            f"- Correos acumulados con intención de inscripción: {total_obs:.0f}\n"
            f"- Proyección ARIMA a 48h: {total_est:.0f} (incremento esperado: +{incremento:.0f})\n"
            f"- Actualmente {int(total_obs)} contactos únicos han manifestado intención de inscribirse; "
            f"para alcanzar la meta de 1,000 inscritos, la tasa de conversión tendría que ser de {tasa_requerida:.1f}%.\n"
            "Sé directo y neutral. No uses listas, solo prosa. "
            "Evita adjetivos calificativos (como 'significativo', 'notable', 'alto', 'bajo', etc.). "
            "Limítate a enunciar los resultados sin valorarlos. "
            "No incluyas título ni encabezado, solo el párrafo."
        )}]
    )
    return msg.content[0].text

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total interesados", total)
c2.metric("Proceso de inscripción", inscripcion, f"{inscripcion/total*100:.1f}%" if total else "—")
c3.metric("Solicitar información", info, f"{info/total*100:.1f}%" if total else "—")
c4.metric("Hombres", hombres, f"{hombres/total*100:.1f}%" if total else "—")
c5.metric("Mujeres", mujeres, f"{mujeres/total*100:.1f}%" if total else "—")
c6.metric("No determinado", indeterminado, f"{indeterminado/total*100:.1f}%" if total else "—")

st.divider()

# ── Leads por hora ────────────────────────────────────────────────────

st.subheader("Interesados por hora")

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
    yaxis_title="Interesados",
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

# ── Penetración de población potencial ───────────────────────────────

POBLACION_POTENCIAL = 462_592

st.subheader("Tasa de contacto única")

por_hora_pen = (
    df_f.set_index('created_at')
    .resample('h')
    .size()
    .rename('registros')
    .to_frame()
)
por_hora_pen['acumulado'] = por_hora_pen['registros'].cumsum()
por_hora_pen['penetracion'] = por_hora_pen['acumulado'] / POBLACION_POTENCIAL * 100

pen_actual = por_hora_pen['penetracion'].iloc[-1]

cp1, cp2 = st.columns(2)
cp1.metric("Registros acumulados", f"{int(por_hora_pen['acumulado'].iloc[-1]):,}")
cp2.metric("Tasa de contacto única", f"{pen_actual:.3f}%")

fig_pen = go.Figure(go.Scatter(
    x=por_hora_pen.index,
    y=por_hora_pen['penetracion'],
    mode='lines+markers',
    line=dict(color='mediumseagreen', width=2),
    marker=dict(size=5),
    hovertemplate='%{x}<br>%{y:.4f}%<extra></extra>',
))
fig_pen.update_layout(
    xaxis_title="Hora",
    yaxis_title="Tasa de contacto única (%)",
    height=360,
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_pen, use_container_width=True)

st.caption(
    f"Población potencial: {POBLACION_POTENCIAL:,} afiliados · Fórmula: (registros acumulados / {POBLACION_POTENCIAL:,}) × 100  \n"
    "**Nota:** La tasa se calcula sobre registros únicos."
)

st.divider()

# ── Proyección ARIMA ──────────────────────────────────────────────────

st.subheader("Interesados únicos acumulados — ARIMA(1,1,0)")

df_arima = df_f[df_f['interesado_en'] == 'Iniciar Proceso de inscripción'].copy()
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

tasa_req = (1_000 / total_obs * 100) if total_obs else 0

with resumen_placeholder.container():
    with st.spinner("Generando resumen..."):
        resumen = generar_resumen_ia(
            total, inscripcion, info, hombres, mujeres,
            fecha_inicio, fecha_fin, total_obs, total_est, incremento, pen_actual, tasa_req,
        )
    st.info(f"### **Resumen ejecutivo ({fmt_fecha(fecha_inicio)} — {fmt_fecha(fecha_fin)})**\n\n{resumen}")

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

st.divider()

# ── Escenarios de conversión ──────────────────────────────────────────

META = 1_000

st.subheader("Escenarios de conversión — meta: 1,000 inscritos")

st.info(
    "**¿Cómo se calcula la tasa de conversión?**  \n"
    "Se toma el total de contactos únicos que manifestaron intención de inscripción y se multiplica "
    "por una tasa asumida para estimar cuántos terminarían inscribiéndose:  \n"
    "**Inscritos estimados = contactos únicos con intención × tasa de conversión**  \n"
    "La tasa requerida es el valor mínimo que necesitaría alcanzarse para llegar a la meta de 1,000 inscritos."
)

_, cm, _ = st.columns([1, 1, 1])
cm.metric(
    "Tasa de conversión requerida para alcanzar la meta",
    f"{tasa_req:.1f}%",
)

tasa_slider = st.slider("Tasa de conversión (%)", 1, 100, 25)
inscritos_slider = total_obs * tasa_slider / 100
color_barra = 'mediumseagreen' if inscritos_slider >= META else 'steelblue'

fig_esc = go.Figure(go.Bar(
    y=[f"{tasa_slider}%"],
    x=[inscritos_slider],
    orientation='h',
    marker_color=color_barra,
    text=[f"{inscritos_slider:.0f}"],
    textposition='outside',
))
fig_esc.add_vline(
    x=META,
    line_dash='dash',
    line_color='gold',
    annotation_text=f"Meta: {META:,}",
    annotation_position="top",
)
fig_esc.update_layout(
    xaxis_title="Inscritos estimados",
    xaxis=dict(range=[0, max(inscritos_slider * 1.2, META * 1.1)]),
    height=220,
    margin=dict(t=30, b=10, r=80),
    showlegend=False,
)
st.plotly_chart(fig_esc, use_container_width=True)

st.caption(
    f"Basado en {int(total_obs):,} contactos únicos con intención de inscripción. "
    "No representan inscripciones confirmadas."
)

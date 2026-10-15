import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Detector de Sono Gamer", page_icon="🎮", layout="wide")

st.title("🎮 Monitor Interativo de Sono e Fadiga Gamer")
st.caption("Gráficos dinâmicos acoplados à inferência em tempo real da rede neural TensorFlow.")

# 1. Pipeline de Treino (em cache para não travar a interface)
@st.cache_resource
def treinar_modelo():
    X = np.array([[1.0], [2.0], [4.0], [6.0], [8.0], [10.0]], dtype=float)
    y = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0], dtype=float)

    net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(units=8, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mse')
    net.fit(X, y, epochs=250, verbose=0)
    return net

modelo = treinar_modelo()

# 2. Entrada do Usuário
st.sidebar.header("🕹️ Painel de Controle")
horas_input = st.sidebar.number_input(
    label="Quantas horas você passou jogando?",
    min_value=0.0,
    max_value=14.0,
    value=5.0,
    step=0.5
)

# 3. Predição com a Rede Neural
entrada = np.array([[float(horas_input)]])
fadiga_prevista = float(modelo.predict(entrada, verbose=0)[0][0])
fadiga_prevista = float(np.clip(fadiga_prevista, 0.0, 10.0))

# 4. Painel de Status
col_diag, col_metric = st.columns([2, 1])
with col_metric:
    st.metric(
        label="Nível de Cansaço Previsto",
        value=f"{fadiga_prevista:.1f} / 10",
        delta=f"+{fadiga_prevista:.1f}" if fadiga_prevista > 5 else "Estável",
        delta_color="inverse"
    )

with col_diag:
    if fadiga_prevista >= 7.5:
        st.error(f"🚨 **Zona Crítica ({horas_input}h jogadas):** Privação de sono e reflexos comprometidos. Desligue a tela imediatamente.")
    elif fadiga_prevista >= 4.5:
        st.warning(f"⚠️ **Zona de Atenção ({horas_input}h jogadas):** Cansaço acumulado. Faça uma pausa de 15 minutos.")
    else:
        st.success(f"⚡ **Zona Segura ({horas_input}h jogadas):** Nível de foco e energia preservados.")

st.divider()

# 5. Gráficos 100% Dinâmicos
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("📈 Sua Posição na Curva de Fadiga")
    st.caption("Acompanhe o marcador de cansaço se deslocando conforme você altera as horas.")

    # Cria pontos de 0 até o limite máximo incluindo EXATAMENTE o ponto do usuário
    pontos_horas = sorted(list(set(list(np.linspace(0, 14, 29)) + [horas_input])))
    preds = modelo.predict(np.array(pontos_horas).reshape(-1, 1), verbose=0).flatten()
    preds = np.clip(preds, 0.0, 10.0)

    # Cria série onde apenas o ponto do usuário recebe o valor (destaque visual dinâmico)
    ponto_atual = [fadiga_prevista if round(h, 2) == round(horas_input, 2) else None for h in pontos_horas]

    df_curva = pd.DataFrame({
        "Horas de Jogo": [round(h, 2) for h in pontos_horas],
        "Curva da IA (Geral)": preds,
        "Você Está Aqui": ponto_atual
    }).set_index("Horas de Jogo")

    st.line_chart(df_curva)

with col_graf2:
    st.subheader("📊 Queda de Rendimento em Tempo Real")
    st.caption("Compare a perda de reflexo, mira e foco contra o estado 100% descansado.")

    # Degradação calculada com base na fadiga predita pela IA
    foco = max(5, int(100 - (fadiga_prevista * 8.5)))
    mira = max(5, int(100 - (fadiga_prevista * 9.0)))
    reflexo = max(5, int(100 - (fadiga_prevista * 8.0)))

    df_performance = pd.DataFrame({
        "Métricas": ["Foco Mental", "Precisão / Mira", "Tempo de Reação"],
        "Descansado (Base)": [100, 100, 100],
        "Seu Estado Atual": [foco, mira, reflexo]
    }).set_index("Métricas")

    st.bar_chart(df_performance)
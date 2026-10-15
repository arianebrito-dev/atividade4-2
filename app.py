import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Detector de Sono Gamer", page_icon="🎮", layout="wide")

st.title("🎮 Conscientização e Detector de Fadiga Gamer")
st.caption("Modelo em TensorFlow/Keras analisando o impacto do tempo excessivo de tela no corpo e nos reflexos.")

# 1. Pipeline de Treinamento da Rede Neural
@st.cache_resource
def treinar_modelo():
    # Dados de treino: horas seguidas jogando vs escala de fadiga (1 a 10)
    X = np.array([[1.0], [2.0], [4.0], [6.0], [8.0], [10.0]], dtype=float)
    y = np.array([2.0, 5.0, 9.0, 12.0, 15.0, 18.0], dtype=float)

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
st.sidebar.header("🕹️ Painel do Jogador")
horas_input = st.sidebar.number_input(
    label="Quantas horas contínuas você passou jogando?",
    min_value=0.0,
    max_value=16.0,
    value=4.0,
    step=0.5
)

# Predição do cansaço com a rede neural
entrada = np.array([[horas_input]])
fadiga_prevista = float(modelo.predict(entrada, verbose=0)[0][0])
fadiga_prevista = float(np.clip(fadiga_prevista, 0.0, 10.0))

# 3. Diagnóstico e Feedback Educativo
col_diag, col_metric = st.columns([2, 1])

with col_metric:
    st.metric("Nível de Cansaço Previsto", f"{fadiga_prevista:.1f} / 10")
    
with col_diag:
    if fadiga_prevista >= 7.5:
        st.error("🚨 **Zona Crítica de Exaustão:** A privação de descanso afeta o córtex pré-frontal, degradando drasticamente tomada de decisão e reflexos. Encerre o gameplay.")
    elif fadiga_prevista >= 4.5:
        st.warning("⚠️ **Zona de Alerta:** A fadiga ocular e muscular começa a acumular. Uma pausa de 20 minutos com hidratação é essencial.")
    else:
        st.success("⚡ **Zona Segura:** Estado de alerta e tempo de reação mantidos dentro da faixa ideal.")

st.divider()

# 4. Gráficos de Conscientização
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("📈 Curva de Crescimento da Fadiga (IA)")
    st.caption("Note como o cansaço acelera de forma agressiva após 6 horas contínuas.")
    
    # Gera a curva predita pela rede de 0 a 12 horas
    horas_range = np.linspace(0, 12, 25)
    predicoes = modelo.predict(horas_range.reshape(-1, 1), verbose=0).flatten()
    predicoes = np.clip(predicoes, 0, 10)
    
    df_curva = pd.DataFrame({
        "Horas de Tela": horas_range,
        "Cansaço Previsto": predicoes
    }).set_index("Horas de Tela")
    
    st.line_chart(df_curva)

with col_graf2:
    st.subheader("📊 Degradação de Habilidade Gamer (%)")
    st.caption("Impacto fisiológico estimado com base na sua sessão atual.")
    
    # Cálculos demonstrativos baseados na exaustão prevista
    foco = max(10, int(100 - (fadiga_prevista * 8.5)))
    mira_precisao = max(10, int(100 - (fadiga_prevista * 9.0)))
    reflexos = max(10, int(100 - (fadiga_prevista * 8.0)))
    
    df_habilidades = pd.DataFrame({
        "Atributo": ["Foco Mental", "Precisão / Mira", "Velocidade de Reflexo"],
        "Performance Atual (%)": [foco, mira_precisao, reflexos]
    }).set_index("Atributo")
    
    st.bar_chart(df_habilidades)




# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LinearRegression

# st.title("🎮 Detector de Sono Gamer")

# # Dados de treino
# gamer = pd.DataFrame({
#     'horas_jogo': [1, 2, 4, 6, 8, 10],
#     'cansaco': [1, 2, 3, 5, 8, 10]
# })

# # Treinamento do modelo
# X = gamer[['horas_jogo']]
# y = gamer['cansaco']
# modelo = LinearRegression().fit(X, y)

# # Entrada do usuário
# horas = st.slider("Quantas horas você passou jogando?", min_value=0, max_value=24, value=5)

# # Predição
# estimativa = modelo.predict(pd.DataFrame({'horas_jogo': [horas]}))[0]

# # Exibição do resultado
# st.subheader("Resultado:")
# st.write(f"Nível de cansaço estimado: **{estimativa:.1f} / 10**")

# if estimativa >= 7:
#     st.warning("⚠️ Hora de pausar e descansar!")
# else:
#     st.success("✅ Energia suficiente para continuar.")
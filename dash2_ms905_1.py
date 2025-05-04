import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import chi2
#import random
st.markdown("""
<style>
.tit {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

#st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

# ========================
# 1. Carregamento de Dados
# ========================
file_path = r".\heart.csv"
df = pd.read_csv(file_path)

# Modificações no df

sex_map = {'M': 0, 'F': 1}
df['Sex'] = df['Sex'].map(sex_map)

df['RestingECG_ST'] = (df['RestingECG'] == 'ST').astype(int)
df['RestingECG_LVH'] = (df['RestingECG'] == 'LVH').astype(int)
df['RestingECG_N'] = (df['RestingECG'] == 'Normal').astype(int)
df = df.drop(columns=['RestingECG'])

df['ChestPainType_TA'] = (df['ChestPainType'] == 'TA').astype(int)
df['ChestPainType_ATA'] = (df['ChestPainType'] == 'ATA').astype(int)
df['ChestPainType_NAP'] = (df['ChestPainType'] == 'NAP').astype(int)
df['ChestPainType_ASY'] = (df['ChestPainType'] == 'ASY').astype(int)
df = df.drop(columns=['ChestPainType'])

df['ExerciseAngina'] = (df['ExerciseAngina'] == 'Y').astype(int)

st_slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}
df['ST_Slope'] = df['ST_Slope'].map(st_slope_map)

df_trad = df.rename(
    columns={'Age': 'Idade', 'Sex': 'Gênero', 'Cholesterol': 'Colesterol', 'HeartDisease': 'Doença cardíaca',
             'RestingBP': 'Pressão repouso', 'FastingBS': 'Diabetes', 'MaxHR': 'Máxima frequência cardíaca',
             'RestingECG_ST': 'ECG de repouso com ST anormal', 'RestingECG_LVH': 'ECG de repouso com HVE',
             'RestingECG_N': 'ECG de repouso normal', 'ExerciseAngina': 'Dor no peito via exercício',
             'ChestPainType_TA': 'Dor no peito típica', 'ChestPainType_ATA': 'Dor no peito não típica',
             'ChestPainType_NAP': 'Outras dores', 'ChestPainType_ASY': 'Sem dores', 'ST_Slope': 'Ângulo ST',
             'Oldpeak': 'ST'})


# ========================
# 2. Funções
# ========================

def get_filled_cholesterol_df():
    df_trad_copy = df_trad.copy()
    # 1. Select continuous features (except Colesterol)
    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST']
    df_cont = df_trad_copy[cont_features]

    # 2. Scale continuous features
    scaler = StandardScaler()
    df_cont_scaled = pd.DataFrame(scaler.fit_transform(df_cont), columns=cont_features)

    # 3. Combine scaled continuous features back with rest of df (excluding Colesterol)
    cholesterol = df_trad_copy['Colesterol']  # save for coloring later
    df_pca_base = pd.concat([df_trad_copy.drop(columns=cont_features + ['Colesterol']), df_cont_scaled], axis=1)

    # 4. Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_pca_base)
    df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])

    # 5. Add Colesterol back for coloring
    df_pca['Colesterol'] = cholesterol

    df_known = df_pca[df_pca['Colesterol'] > 0].copy()
    df_missing = df_pca[df_pca['Colesterol'] == 0].copy()

    # For each missing value
    for idx, row in df_missing.iterrows():
        point = np.array([[row['PC1'], row['PC2']]])  # Shape (1, 2)

        # Get distances to all known points
        known_points = df_known[['PC1', 'PC2']].values
        distances = euclidean_distances(point, known_points).flatten()

        # Find 3 nearest neighbors
        nearest_idxs = distances.argsort()[:3]
        nearest_chol_vals = df_known.iloc[nearest_idxs]['Colesterol']

        # Mean of neighbors
        imputed_value = nearest_chol_vals.mean()

        # Update original df (df_pca)
        df_pca.at[idx, 'Colesterol'] = imputed_value

    df_final = df_trad.copy()
    df_final = df_trad.drop(columns=['Colesterol'])
    df_final = pd.concat([df_final, df_pca[['Colesterol']]], axis=1)

    return df_final


def get_cantour():
    df_trad_10 = get_filled_cholesterol_df()

    # 0. aqui, vamos dropar o que vimos no teste chi2 que tem p-value >0.05
    df_trad_simp = df_trad_10.copy().drop(
        columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE', 'Doença cardíaca'])

    # 1. Select continuous features (except Colesterol)
    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST', 'Colesterol']
    df_cont = df_trad_simp[cont_features].copy()

    # 2. Scale continuous features
    scaler = StandardScaler()
    df_cont_scaled = pd.DataFrame(scaler.fit_transform(df_cont), columns=cont_features)

    # 3. Combine scaled continuous features back with rest of df
    df_scaled = pd.concat([df_trad_simp.drop(columns=cont_features), df_cont_scaled], axis=1)

    pca_final2 = PCA(n_components=0.95)
    pca_components_final = pca_final2.fit_transform(df_scaled)

    df_pca_final2 = pd.DataFrame(pca_components_final
                                , columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
                                )

    # 7. Add back the heart disease column and map for color
    df_pca_final2['Doença cardíaca'] = df_trad['Doença cardíaca'].copy().reset_index(drop=True)
    df_pca_final2['Diagnóstico'] = df_pca_final2['Doença cardíaca'].map({1: 'Paciente', 0: 'Saudável'})

    pacientes = df_pca_final2[df_pca_final2['Doença cardíaca'] == 1][['PC1', 'PC2']].values
    saudaveis = df_pca_final2[df_pca_final2['Doença cardíaca'] == 0][['PC1', 'PC2']].values

    gmm_pacientes = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(pacientes)
    gmm_saudaveis = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(saudaveis)

    X_all = df_pca_final2[['PC1', 'PC2']].values

    log_likelihood_pacientes = gmm_pacientes.score_samples(X_all)
    log_likelihood_saudaveis = gmm_saudaveis.score_samples(X_all)

    likelihood_pacientes = np.exp(log_likelihood_pacientes)
    likelihood_saudaveis = np.exp(log_likelihood_saudaveis)

    posterior_paciente = likelihood_pacientes / (likelihood_pacientes + likelihood_saudaveis)

    df_pca_final2['Prob_Paciente'] = posterior_paciente

    # Define the grid range
    x_min, x_max = df_pca_final2['PC1'].min() - 1, df_pca_final2['PC1'].max() + 1
    y_min, y_max = df_pca_final2['PC2'].min() - 1, df_pca_final2['PC2'].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    log_likelihood_p = gmm_pacientes.score_samples(grid_points)
    log_likelihood_s = gmm_saudaveis.score_samples(grid_points)

    likelihood_p = np.exp(log_likelihood_p)
    likelihood_s = np.exp(log_likelihood_s)

    posterior_grid = likelihood_p / (likelihood_p + likelihood_s)
    posterior_grid = posterior_grid.reshape(xx.shape)

    return posterior_grid


# ========================
# 2. Título e Menu Interativo
# ========================
st.header("Análise de Dados - Presença de Doenças Cardíacas")

# Menu de seleção de visualizações
# Initialize view state
#if "view_option" not in st.session_state:
 #   st.session_state.view_option = None

#st.markdown("### Escolha a visualização:")

#col1, col2, col4, col5, col6, col7, col8, col9 = st.columns(8)

#with col1:
 #   if st.button("Tabela de Dados"):
  #      st.session_state.view_option = "Tabela de Dados"
#with col2:
 #   if st.button("Distribuição demográfica"):
  #      st.session_state.view_option = "Distribuição de Idade por Sexo"
# with col3:
#   if st.button("🎂 Idade x Sexo"):
#      st.session_state.view_option = "Distribuição de Idade por Sexo"
#with col4:
 #   if st.button("Correlação"):
  #      st.session_state.view_option = "Matriz de Correlação"
#with col5:
 #   if st.button("Visualização de Pacientes"):
  #      st.session_state.view_option = "Visualização de Pacientes"
#with col6:
 #   if st.button("Qui-Quadrado"):
  #      st.session_state.view_option = "Qui-Quadrado"
#with col7:
 #   if st.button("Atualizando valores de Colesterol"):
  #      st.session_state.view_option = "PCA sem Colesterol"
#with col8:
 #   if st.button("PCA e Clusterização"):
  #      st.session_state.view_option = "PCA"
#with col9:
 #   if st.button("Simulação"):
  #      st.session_state.view_option = "Simulação"

# Now this value persists across reruns
#opcao = st.session_state.view_option

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Tabela de Dados","Distribuição demográfica",
                                                          "Visualização de Pacientes","Teste de Qui-Quadrado","Matriz de Correlação",
                                                           "Atualizando valores de Colesterol"
                                                          ,"PCA e Clusterização","Simulação Interativa na PCA"])
# ========================
# 3. Visualizações com base na seleção
# ========================

#if opcao == "Tabela de Dados":
with tab1:
    col61, col62, col63 =st.columns([0.14, 0.26, 0.60])
    with col61:
        st.write('Número de entradas: '+ str(len(df_trad))+'.')
    with col63:
        st.write('Linhas deletadas: 1 (Pressão de repouso zerada, Colesterol zerado).')
    with col62:
        df_col_01 = df_trad.copy()
        df_col_01 = df_col_01[df_col_01['Colesterol'] == 0]
        st.write('Linhas com falta de informação no Colesterol: '+str(len(df_col_01))+'.')
    df_trad = df_trad[df_trad['Pressão repouso'] > 0]
    df_trad=df_trad.reset_index().drop(columns=['index'])
    st.dataframe(df_trad)


# ==========================================================================================================================================================================

#elif opcao == "Distribuição de Idade por Sexo":
with tab2:
   # st.subheader("Distribuição por Idade, Gênero e Diagnóstico")
    sex_label_map = {0: 'Masculino', 1: 'Feminino'}
    df_age = df.groupby(by=['Age', 'Sex']).size().reset_index(name='count')
    df_age['Sex'] = df_age['Sex'].map(sex_label_map)
    df_sex = df['Sex'].value_counts().reset_index(name='count')

    df_sex['Sex'] = df_sex['Sex'].map(sex_label_map)

    fig1 = px.pie(df_sex, values='count', names='Sex')
    fig1.update_traces(textfont=dict(size=30))
    fig1.update_layout(legend_font=dict(size=20))
    fig2 = px.bar(df_age, x='Age', y='count', color='Sex')
    fig2.update_layout(showlegend=False,
                       xaxis_title="Idade",
                       yaxis_title="Contagem"
                       )
    df_doente = df['HeartDisease'].value_counts().reset_index(name='Contagem')
    df_doente = df_doente.replace({1: 'Paciente', 0: 'Saudável'})

    fig3 = px.pie(df_doente, values='Contagem', names='HeartDisease'
                  , color_discrete_sequence=['red','green']
                  )
    fig3.update_traces(textfont=dict(size=30),
                       texttemplate=' %{percent:.0%}')
    fig3.update_layout(legend_font=dict(size=20),
                       #     template = 'plotly_white',
                       legend_title=None)

    col6, col7, col8 = st.columns([0.44,0.28,0.28])
    with col6:
        st.plotly_chart(fig2, use_container_width=True)
    with col7:
        st.plotly_chart(fig1, use_container_width=True)
    with col8:
        st.plotly_chart(fig3, use_container_width=True)


# ==========================================================================================================================================================================

#elif opcao == "Matriz de Correlação":
with tab5:
    #st.subheader("Matriz de Correlação")

    df_trad_corr = df_trad.copy().drop(columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE'])
    numeric_df = df_trad_corr.select_dtypes(include=np.number)
    #st.write(numeric_df)

    # Calcula a matriz de correlação
    corr_matrix = numeric_df.corr()
    # st.dataframe(corr_matrix)
    # Garantir que os valores estão no intervalo [-1, 1]
    corr_matrix = corr_matrix.round(2)
    # st.dataframe(corr_matrix)
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='icefire_r', range_color=(-1, 1))
    # fig_corr, ax = plt.subplots(figsize=(10, jet hot rainbow thermal solar
    # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    # plt.title("Matriz de Correlação", fontsize=16)
    # st.pyplot(fig_corr)
    fig_corr.update_layout(width=1000,height=600,legend=dict(xanchor='left'))
    st.plotly_chart(fig_corr, use_container_width=True)

    numeric_df = df_trad.select_dtypes(include=np.number)


# ==========================================================================================================================================================================

#elif opcao == "Qui-Quadrado":
with tab4:
   # st.subheader("Qui-Quadrado entre variáveis categóricas e presença de Doença Cardíaca")

    df_chi = df_trad.copy()

    # Target must be binary integer
    y = df_chi['Doença cardíaca']

    # Select categorical features (assume those with few unique values)
    cat_features = df_chi.select_dtypes(include=['int64', 'int32']).nunique()
    cat_features = cat_features[(cat_features > 1) & (cat_features < 10)].index.tolist()
    cat_features.remove('Doença cardíaca')

    # Run chi-squared test
    chi_scores, p_values = chi2(df_chi[cat_features], y)

    # Show results in DataFrame
    chi_df = pd.DataFrame({
        'Variável': cat_features,
        'Chi²': chi_scores,
        'p-valor': p_values
    }).sort_values(by='Chi²', ascending=False)

    st.dataframe(chi_df.style.format({'Chi²': '{:.2f}', 'p-valor': '{:.4f}'}))



# ==========================================================================================================================================================================

#elif opcao == "Visualização de Pacientes":
with tab3:
   # st.subheader('Dispersão dos dados coloridos pela presença de Doença Cardíaca')
    col10, col11 = st.columns(2)
    with col10:
            x_column = st.selectbox("Selecione o eixo X do gráfico:", df_trad.columns, index=3)
    with col11:
            y_column = st.selectbox("Selecione o eixo Y do gráfico:", df_trad.columns, index=5)

    df_trad1s = df_trad.copy()
    df_trad1s['Diagnóstico'] = df_trad1s['Doença cardíaca'].map({1: 'Paciente', 0: 'Saudável'})

    fig1s = px.scatter(
        df_trad1s,
        x=x_column,
        y=y_column,
        color='Diagnóstico',
        color_discrete_map={'Paciente': 'red', 'Saudável': 'green'},
        labels={'Diagnóstico': 'Diagnóstico'},
        title = 'Dispersão dos dados coloridos pela presença de Doença Cardíaca'
    )
    fig1s.update_layout(height = 400)
    st.plotly_chart(fig1s)



# ==========================================================================================================================================================================

#elif opcao == "PCA sem Colesterol":
with tab6:
    # 0. aqui, vamos dropar o que vimos no teste chi2 que tem p-value >0.05

    df_trad_simp2 = df_trad.drop(columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE'])
    # 1. Select continuous features (except Colesterol)

    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST']
    df_cont = df_trad_simp2[cont_features]


    # 2. Scale continuous features
    scaler = StandardScaler()
    df_cont_scaled = pd.DataFrame(scaler.fit_transform(df_cont), columns=cont_features)

    # 3. Combine scaled continuous features back with rest of df (excluding Colesterol)
    cholesterol = df_trad_simp2['Colesterol']  # save for coloring later

    df_pca_base2 = pd.concat([df_trad_simp2.drop(columns=cont_features + ['Colesterol']), df_cont_scaled], axis=1)

    # 4. Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_pca_base2)
    df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])

    # 5. Add Colesterol back for coloring
    df_pca['Colesterol'] = cholesterol

    # 6. Plot
    fig_pca = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='Colesterol',
        color_continuous_scale='purd',
        labels={'Colesterol': 'Colesterol'},
        range_color=(0, 600),
        title = 'PCA colorido pelos valores originais de Colesterol'
    )
    #st.plotly_chart(fig_pca)

    df_known = df_pca[df_pca['Colesterol'] > 0].copy()
    df_missing = df_pca[df_pca['Colesterol'] == 0].copy()

    # For each missing value
    for idx, row in df_missing.iterrows():
        point = np.array([[row['PC1'], row['PC2']]])  # Shape (1, 2)

        # Get distances to all known points
        known_points = df_known[['PC1', 'PC2']].values
        distances = euclidean_distances(point, known_points).flatten()

        # Find 3 nearest neighbors
        nearest_idxs = distances.argsort()[:3]
        nearest_chol_vals = df_known.iloc[nearest_idxs]['Colesterol']

        # Mean of neighbors
        imputed_value = nearest_chol_vals.mean()

        # Update original df (df_pca)
        df_pca.at[idx, 'Colesterol'] = imputed_value

    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='Colesterol',
        color_continuous_scale='purd',
        title="PCA colorido pelos valores atualizados de Colesterol",
        range_color=(0, 600)
    )
    col21, col22 = st.columns([0.46, 0.54])
    fig_pca.update_layout(coloraxis_showscale =False,
                          height = 500)
    fig.update_layout(height = 500)

    with col21:
        st.plotly_chart(fig_pca)
    with col22:
        st.plotly_chart(fig)



# ==========================================================================================================================================================================

#elif opcao == "PCA":
with tab7:
    st.subheader('Visualizações do PCA colorido pela presença de Doença Cardíaca')
    df_trad_2 = get_filled_cholesterol_df().copy()

    # 0. aqui, vamos dropar o que vimos no teste chi2 que tem p-value >0.05
    df_trad_simp3 = df_trad_2.drop(
        columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE', 'Doença cardíaca']).copy()

    # 1. Select continuous features (except Colesterol)
    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST', 'Colesterol']
    df_cont = df_trad_simp3[cont_features].copy()

    # 2. Scale continuous features
    scaler = StandardScaler()
    df_cont_scaled = pd.DataFrame(scaler.fit_transform(df_cont), columns=cont_features)

    # 3. Combine scaled continuous features back with rest of df
    df_scaled = pd.concat([df_trad_simp3.drop(columns=cont_features), df_cont_scaled], axis=1)

    pca_final = PCA(n_components=0.95)
    pca_components_final = pca_final.fit_transform(df_scaled)

    df_pca_final = pd.DataFrame(pca_components_final
                                , columns=['PC1'
            , 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'
                                           ]
                                )

    # 7. Add back the heart disease column and map for color
    df_pca_final['Doença cardíaca'] = df_trad['Doença cardíaca'].copy().reset_index(drop=True)
    df_pca_final['Diagnóstico'] = df_pca_final['Doença cardíaca'].copy().map({1: 'Paciente', 0: 'Saudável'})

    # 8. Plot PCA colored by heart disease
    fig3d = px.scatter_3d(
        df_pca_final,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Diagnóstico',
        color_discrete_map={'Paciente': 'red', 'Saudável': 'green'}
        ,title='3 PCs: 58% da variância explicada'
    )
    fig3d.update_layout(
        showlegend = False
        #,width=1500,
        #height=1000
    )

    fig2d = px.scatter(
        df_pca_final,
        x='PC1',
        y='PC2',
        color='Diagnóstico',
        color_discrete_map={'Paciente': 'red', 'Saudável': 'green'},
        title='2 PCs: 45% da variância explicada'
    )

    col31, col32 = st.columns(2)
    with col31:
        st.plotly_chart(fig3d)
    with col32:
        st.plotly_chart(fig2d)

    explained_variance = pca_final.explained_variance_ratio_.copy()

#    st.markdown('<p class="tit">**Variância** explicada pelos Componentes Principais e sua Composição**</p>',unsafe_allow_html=True)
    st.subheader('Variância explicada pelos Componentes Principais e sua Composição')
    fig_var = px.bar(
        x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'],
        y=explained_variance[:10]*100,
        labels={'x': 'Componente Principal', 'y': 'Variância Explicada (%)'}
        #,title='Variância Explicada por PCA'
        ,color_discrete_sequence = [ '#AB63FA']
    )
    #st.plotly_chart(fig_var)

    explained_variance = pca_final.explained_variance_ratio_.copy()

    # Plot feature composition (loadings)
    loadings = pd.DataFrame(pca_final.components_.T, index=df_scaled.columns,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

    explained_var = pca_final.explained_variance_ratio_.copy()
    col41, col42 = st.columns(2)
    with col41:
        st.plotly_chart(fig_var)
    with col42:
      #  st.write("Representatividade das Componentes Principais:")
       # for i, var in enumerate(explained_var):
        #    st.write(f"PC{i + 1}: {var:.2%}")
       # st.write("Composição das Componentes Principais (Loadings)")
        st.dataframe(loadings.style.background_gradient(cmap='coolwarm_r', axis=0))
    #color_continuous_scale = 'icefire_r', range_color = (-1, 1)
    pacientes = df_pca_final[df_pca_final['Doença cardíaca'] == 1][['PC1', 'PC2']].values
    saudaveis = df_pca_final[df_pca_final['Doença cardíaca'] == 0][['PC1', 'PC2']].values

    gmm_pacientes = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(pacientes)
    gmm_saudaveis = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(saudaveis)

    X_all = df_pca_final[['PC1', 'PC2']].values

    log_likelihood_pacientes = gmm_pacientes.score_samples(X_all)
    log_likelihood_saudaveis = gmm_saudaveis.score_samples(X_all)

    likelihood_pacientes = np.exp(log_likelihood_pacientes)
    likelihood_saudaveis = np.exp(log_likelihood_saudaveis)

    posterior_paciente = likelihood_pacientes / (likelihood_pacientes + likelihood_saudaveis)

    df_pca_final['Prob_Paciente'] = posterior_paciente.copy()

    fig = px.scatter(
        df_pca_final,
        x='PC1',
        y='PC2',
        color='Prob_Paciente',
        color_continuous_scale='rdylgn_r',
        title='Probabilidade de Ser Paciente (via GMM + PCA)',
        labels={'Prob_Paciente': 'P(Paciente)'}
    )
    fig.update_layout(coloraxis_showscale = False)


    # Define the grid range
    x_min, x_max = df_pca_final['PC1'].min() - 1, df_pca_final['PC1'].max() + 1
    y_min, y_max = df_pca_final['PC2'].min() - 1, df_pca_final['PC2'].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    log_likelihood_p = gmm_pacientes.score_samples(grid_points)
    log_likelihood_s = gmm_saudaveis.score_samples(grid_points)

    likelihood_p = np.exp(log_likelihood_p)
    likelihood_s = np.exp(log_likelihood_s)

    posterior_grid = likelihood_p / (likelihood_p + likelihood_s)
    posterior_grid = posterior_grid.reshape(xx.shape)

    contour_fig = go.Figure(data=
    go.Contour(
        z=posterior_grid,
        x=np.linspace(x_min, x_max, 200),  # x-axis values
        y=np.linspace(y_min, y_max, 200),  # y-axis values
        colorscale='rdylgn_r',
        contours_coloring='heatmap',
        colorbar=dict(title='P(Paciente)'),
        showscale=True
    )
    )

    contour_fig.update_layout(
        title='Contorno de Probabilidade de Ser Paciente (GMM + PCA)',
        xaxis_title='PC1',
        yaxis_title='PC2'
    )
    st.subheader('Clusterização GMM')
    col51, col52= st.columns(2)
    with col51:
        st.plotly_chart(fig)
    with col52:
        st.plotly_chart(contour_fig)

# ==========================================================================================================================================================================


#if opcao == "Simulação":
with tab8:
    #st.subheader("Simulação Interativa na PCA")

    # Step 0: Use df_trad (with cholesterol imputed) as base
    df_inicial = get_filled_cholesterol_df()

    df_sim = df_inicial.drop(columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE', 'Doença cardíaca'])

    # Step 1: Select continuous features for scaling
    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST', 'Colesterol']
    df_cont = df_sim[cont_features].copy()

    # Step 2: Scale using the same scaler as before
    scaler = StandardScaler()
    df_cont_scaled = pd.DataFrame(scaler.fit_transform(df_cont), columns=cont_features)
    df_scaled = pd.concat([df_sim.drop(columns=cont_features), df_cont_scaled], axis=1)

    # Step 3: Fit PCA and transform
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])

    cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST', 'Colesterol']

    # Create two columns: controls on the left, plot on the right
    col1, col2 = st.columns([1, 3])  # Adjust ratios if needed

    with col1:
        if 'random_idx' not in st.session_state:
            st.session_state.random_idx = None

        if st.button("Selecionar entrada aleatória"):
            st.session_state.random_idx = np.random.randint(len(df_scaled))

        if st.session_state.random_idx is not None:
            selected_feature = st.selectbox("Selecione uma característica para alterar:", cont_features)

            current_val = df_sim.loc[st.session_state.random_idx, selected_feature]
            slider_val = st.slider(
                f"Valor de {selected_feature}",
                float(df_sim[selected_feature].min()),
                float(df_sim[selected_feature].max()),
                float(current_val), step=0.1
            )

    with col2:
        if st.session_state.random_idx is not None:
            # --- existing processing and PCA projection code ---
            row_original = df_sim.loc[[st.session_state.random_idx]].copy()
            row = row_original.copy()
            row.at[st.session_state.random_idx, selected_feature] = slider_val
            row_cont = row[cont_features]
            row = row.drop(columns=cont_features)
            row_cont_scaled = pd.DataFrame(scaler.transform(row_cont), columns=cont_features)
            row_scaled = pd.concat([row.reset_index(drop=True), row_cont_scaled.reset_index(drop=True)], axis=1)

            row_cont = row_original[cont_features]
            row_original = row_original.drop(columns=cont_features)
            row_cont_scaled = pd.DataFrame(scaler.transform(row_cont), columns=cont_features)
            row_original_scaled = pd.concat(
                [row_original.reset_index(drop=True), row_cont_scaled.reset_index(drop=True)], axis=1)

            row_pca = pca.transform(row_scaled)
            row_original_pca = pca.transform(row_original_scaled)

            # ========================
            # Plot
            # ========================

            cantour_config = get_cantour().copy()
            #df_trad_15 = get_filled_cholesterol_df().copy()
            #df_trad_simp3simu = df_trad_15.copy().drop(
             #   columns=['ECG de repouso normal', 'Dor no peito típica', 'ECG de repouso com HVE',
              #           'Doença cardíaca'])

            # 1. Select continuous features (except Colesterol)
            #cont_features = ['Idade', 'Pressão repouso', 'Máxima frequência cardíaca', 'ST', 'Colesterol']
            #df_contsimu = df_trad_simp3simu[cont_features].copy()

            # 2. Scale continuous features
            #scaler = StandardScaler()
            #df_cont_scaledsimu = pd.DataFrame(scaler.fit_transform(df_contsimu), columns=cont_features)

            # 3. Combine scaled continuous features back with rest of df
            #df_scaledsimu = pd.concat([df_trad_simp3simu.drop(columns=cont_features), df_cont_scaledsimu], axis=1)

            #pca_finalsimu = PCA(n_components=0.95)
            #pca_components_finalsimu = pca_finalsimu.fit_transform(df_scaledsimu)

#            df_pca_finalsimu = pd.DataFrame(pca_components_finalsimu
 #                                       , columns=['PC1'
  #                  , 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'
   #                                                ])

            # 7. Add back the heart disease column and map for color
    #        df_pca_finalsimu['Doença cardíaca'] = df_trad['Doença cardíaca'].copy().reset_index(drop=True)
     #       df_pca_finalsimu['Diagnóstico'] = df_pca_finalsimu['Doença cardíaca'].copy().map({1: 'Paciente', 0: 'Saudável'})

       #     pacientessimu = df_pca_finalsimu[df_pca_finalsimu['Doença cardíaca'] == 1][['PC1', 'PC2']].values
      #      saudaveissimu = df_pca_finalsimu[df_pca_finalsimu['Doença cardíaca'] == 0][['PC1', 'PC2']].values

        #    gmm_pacientessimu = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(pacientessimu)
         #   gmm_saudaveissimu = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(saudaveissimu)

          #  X_all = df_pca_finalsimu[['PC1', 'PC2']].values

           # log_likelihood_pacientessimu = gmm_pacientessimu.score_samples(X_all)
            #log_likelihood_saudaveissimu = gmm_saudaveissimu.score_samples(X_all)

            #likelihood_pacientessimu = np.exp(log_likelihood_pacientessimu)
            #likelihood_saudaveissimu = np.exp(log_likelihood_saudaveissimu)

            #posterior_pacientesimu = likelihood_pacientessimu / (likelihood_pacientessimu + likelihood_saudaveissimu)

            #df_pca_finalsimu['Prob_Paciente'] = posterior_pacientesimu.copy()

            fig = go.Figure(data=
            go.Contour(
                z=cantour_config,
                x=np.linspace(-4.5, 4.5, 200),  # x-axis values
                y=np.linspace(-4.5, 4.5, 200),  # y-axis values
                colorscale='rdylgn_r',
                contours_coloring='heatmap',
                colorbar=dict(title='P(Paciente)'),
                showscale=True
            )
            )

            fig.update_layout(
                title='Contorno de Probabilidade de Ser Paciente (GMM + PCA)',
                xaxis_title='PC1',
                yaxis_title='PC2'
            )
	# adding scatter
            fig.add_trace(go.Scatter(
                x=df_pca['PC1'],
                y=df_pca['PC2'],
                mode='markers',
                marker=dict(size=6, color='black', opacity=0.3),
                name='Pontos PCA'
            ))
            fig.add_scatter(x=[row_pca[0, 0]], y=[row_pca[0, 1]], mode='markers',
                            marker=dict(size=12, color='purple', symbol='circle'), name='Entrada Modificada')
            fig.add_scatter(x=[row_original_pca[0, 0]], y=[row_original_pca[0, 1]], mode='markers',
                            marker=dict(size=12, color='#FF0092', symbol='x'), name='Entrada Original')
            fig.update_layout(legend=dict(yanchor="bottom", xanchor="left"))

            st.plotly_chart(fig, use_container_width=True)

#streamlit run C:\Users\55859\PycharmProjects\PythonProject\dash2_ms905_1.py

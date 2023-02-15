import streamlit as st


st.header('Como utilizar')

st.subheader('Através de um arquivo.')

st.markdown('Na aba lateral clique em Predict, após isso escolha a opção de predict **ARQUIVO CSV**, após carregar o modelo irá fazer a previsão do valor e aparecerá um botão de download, o nosso arquivo CSV terá uma última coluna com o nome SalePrice que será a coluna do valor da casa predito pelo modelo.')

st.subheader('Digitando as informações')

st.markdown('Na aba lateral clique em Predict, após isso escolha a opção de predict **INSERINDO OS DADOS**, após preencher os campos com os valores da casa o modelo irá fazer a previsão do valor e disponibilizar o download dos dados da casa junto com o valor no formato .CSV')
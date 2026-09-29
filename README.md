
# Cálculo Interativo — Streamlit

Primeira versão de um site educacional em Python com dois aplicativos:

1. Soma de Riemann
2. Interpretação gráfica da derivada

## Como executar localmente

1. Instale Python 3.10 ou superior.
2. Abra o terminal nesta pasta.
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute:

```bash
streamlit run app.py
```

O navegador deverá abrir automaticamente.

## Publicar no Streamlit Community Cloud

1. Crie um repositório no GitHub.
2. Envie os arquivos desta pasta para o repositório.
3. Acesse https://share.streamlit.io/
4. Entre com sua conta do GitHub.
5. Escolha o repositório.
6. Informe `app.py` como arquivo principal.
7. Clique em Deploy.

## Funções aceitas

A caixa de função aceita, por exemplo:

- `x**2`
- `x^2`
- `x**3 - x`
- `sin(x)`
- `cos(x)`
- `exp(x/3)`
- `log(x)`
- `sqrt(x)`
- `1/(1+x**2)`

## Estrutura

- `app.py` — aplicação principal
- `requirements.txt` — bibliotecas Python
- `.streamlit/config.toml` — configuração visual básica
- `README.md` — instruções

## Próximos módulos sugeridos

- Segunda derivada e concavidade
- Teorema do Valor Médio
- Método de Newton
- Bisseção
- Secante
- Ponto Fixo
- Integração numérica
- Números complexos

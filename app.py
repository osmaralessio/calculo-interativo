
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st

st.set_page_config(
    page_title="Cálculo Interativo",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Estilo ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
.small-note {color: #666; font-size: .92rem;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    padding: 10px 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

x = sp.symbols("x")

# ---------- Parser simples e restrito ----------
ALLOWED_NAMES = {
    "x": x,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "pi": sp.pi, "E": sp.E, "abs": sp.Abs,
}

def parse_function(text):
    text = text.strip().replace("^", "**")
    if not text:
        raise ValueError("Digite uma função.")
    if len(text) > 120:
        raise ValueError("A expressão é muito longa.")
    if re.search(r"[^0-9a-zA-Z_+\-*/().,\s*]", text):
        raise ValueError("A expressão contém caracteres não permitidos.")
    tokens = set(re.findall(r"[A-Za-z_]\w*", text))
    desconhecidos = tokens - set(ALLOWED_NAMES)
    if desconhecidos:
        raise ValueError("Nome(s) não permitido(s): " + ", ".join(sorted(desconhecidos)))
    return sp.sympify(text, locals=ALLOWED_NAMES)

def numpy_function(expr):
    return sp.lambdify(x, expr, modules=["numpy"])

def finite_array(values, base):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(base, float(arr), dtype=float)
    return arr

def adaptive_ylim(ax, *arrays):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [1, 99])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        margin = 0.15 * (hi - lo)
        ax.set_ylim(lo - margin, hi + margin)

# ---------- Navegação ----------
st.sidebar.title("📐 Cálculo Interativo")
pagina = st.sidebar.radio(
    "Navegação",
    ["🏠 Início", "▥ Soma de Riemann", "📈 Derivada gráfica", "🔵 Método de Newton"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Aplicativos interativos para o ensino de Cálculo.")

# ---------- Início ----------
if pagina == "🏠 Início":
    st.title("Cálculo Interativo")
    st.subheader("Visualizações computacionais para o ensino de Cálculo")

    st.write(
        "Esta primeira versão reúne dois aplicativos interativos em Python: "
        "Soma de Riemann e Interpretação Gráfica da Derivada."
    )

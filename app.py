
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
    ["🏠 Início", "▥ Soma de Riemann", "📈 Derivada gráfica"],
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

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "### ▥ Soma de Riemann\n"
            "Explore retângulos pela esquerda, direita ou ponto médio e observe "
            "a aproximação da integral definida."
        )
    with c2:
        st.info(
            "### 📈 Derivada gráfica\n"
            "Observe a reta secante aproximar-se da reta tangente quando "
            "o incremento h tende a zero."
        )

    st.markdown("### Ideia pedagógica")
    st.write(
        "Os controles permitem que o estudante altere parâmetros e veja imediatamente "
        "como as construções geométricas e os valores numéricos se modificam."
    )

    st.markdown("### Próximos módulos possíveis")
    st.write(
        "Segunda derivada e concavidade, Teorema do Valor Médio, Newton, Bisseção, "
        "Secante, Ponto Fixo, integração numérica e números complexos."
    )

# ---------- Riemann ----------
elif pagina == "▥ Soma de Riemann":
    st.title("▥ Soma de Riemann")
    st.write("Aproxime a integral definida por retângulos e compare com o valor da integral.")

    col_controls, col_plot = st.columns([1, 2], gap="large")

    with col_controls:
        funcao_txt = st.text_input("Função f(x)", "4 - 0.6*x + 0.04*x**2", key="r_func")
        a = st.number_input("Limite inferior a", value=0.0, step=0.5, key="r_a")
        b = st.number_input("Limite superior b", value=6.0, step=0.5, key="r_b")
        n = st.slider("Número de retângulos n", 1, 100, 8, key="r_n")
        metodo = st.selectbox("Método", ["Esquerda", "Direita", "Ponto médio"], index=2)
        destaque = st.slider("Retângulo destacado", 1, n, min(1, n), key="r_dest")

    try:
        expr = parse_function(funcao_txt)
        f = numpy_function(expr)
        if b <= a:
            st.error("É necessário ter b > a.")
            st.stop()

        dx = (b - a) / n
        bordas = np.linspace(a, b, n + 1)
        x_esq = bordas[:-1]

        if metodo == "Esquerda":
            xi = x_esq
        elif metodo == "Direita":
            xi = bordas[1:]
        else:
            xi = x_esq + dx / 2

        alturas = finite_array(f(xi), xi)
        soma = float(np.sum(alturas * dx))

        xx = np.linspace(a, b, 1500)
        yy = finite_array(f(xx), xx)

        try:
            integral_exata_expr = sp.integrate(expr, (x, a, b))
            integral_exata = float(sp.N(integral_exata_expr))
        except Exception:
            integral_exata_expr = None
            integral_exata = float(np.trapz(yy, xx))

        erro = abs(integral_exata - soma)

        with col_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(xx, yy, linewidth=3, label="f(x)")
            ax.fill_between(xx, yy, 0, alpha=0.08)

            for i in range(n):
                alpha = 0.75 if i == destaque - 1 else 0.32
                lw = 2.8 if i == destaque - 1 else 1.0
                ax.bar(
                    x_esq[i], alturas[i], width=dx, align="edge",
                    alpha=alpha, edgecolor="black", linewidth=lw, zorder=2
                )
                ax.scatter(xi[i], alturas[i], s=24, zorder=4)

            ax.scatter(xi[destaque - 1], alturas[destaque - 1], s=110, zorder=6)
            ax.axhline(0, linewidth=0.8)
            ax.grid(alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(f"Soma de Riemann — {metodo}")
            ax.legend()
            st.pyplot(fig, clear_figure=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Δx", f"{dx:.6f}")
        m2.metric("Soma Sₙ", f"{soma:.8f}")
        m3.metric("Integral", f"{integral_exata:.8f}")
        m4.metric("Erro absoluto", f"{erro:.3e}")

        i = destaque - 1
        st.latex(r"\Delta x=\frac{b-a}{n}")
        st.latex(r"S_n=\sum_{i=1}^{n} f(x_i^*)\,\Delta x")
        st.write(
            f"Retângulo {destaque}:  x* = {xi[i]:.6f},  "
            f"f(x*) = {alturas[i]:.6f},  área = {alturas[i]*dx:.6f}."
        )

    except Exception as e:
        st.error(f"Não foi possível processar a função: {e}")

# ---------- Derivada ----------
else:
    st.title("📈 Interpretação Gráfica da Derivada")
    st.write("Observe a reta secante aproximar-se da reta tangente quando h tende a zero.")

    col_controls, col_plot = st.columns([1, 2], gap="large")

    with col_controls:
        funcao_txt = st.text_input("Função f(x)", "x**2", key="d_func")
        a = st.slider("Ponto a", -5.0, 5.0, 1.0, 0.1, key="d_a")
        h = st.slider("Incremento h", 0.01, 4.0, 1.0, 0.01, key="d_h")
        animar = st.button("▶ Animar h → 0", use_container_width=True)

    try:
        expr = parse_function(funcao_txt)
        dexpr = sp.diff(expr, x)
        f = numpy_function(expr)
        df = numpy_function(dexpr)

        fa = float(f(a))
        deriv_a = float(df(a))

        def make_derivative_plot(h_value):
            faq = float(f(a + h_value))
            m_sec = (faq - fa) / h_value
            delta_y = faq - fa

            span = max(4.0, 2.2 * abs(h_value) + 2)
            xx = np.linspace(a - span, a + span, 1300)
            yy = finite_array(f(xx), xx)
            ysec = fa + m_sec * (xx - a)
            ytan = fa + deriv_a * (xx - a)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(xx, yy, linewidth=3, label="f(x)")
            ax.plot(xx, ysec, "--", linewidth=2.2, label=f"Secante: m = {m_sec:.6f}")
            ax.plot(xx, ytan, ":", linewidth=2.8, label=f"Tangente: f'(a) = {deriv_a:.6f}")

            ax.scatter([a], [fa], s=100, zorder=6)
            ax.scatter([a+h_value], [faq], s=100, zorder=6)
            ax.annotate("P", (a, fa), xytext=(8, 12), textcoords="offset points", fontsize=14)
            ax.annotate("Q", (a+h_value, faq), xytext=(8, -18), textcoords="offset points", fontsize=14)

            ax.plot([a, a+h_value], [fa, fa], linewidth=2)
            ax.plot([a+h_value, a+h_value], [fa, faq], linewidth=2)

            ax.annotate(r"$\Delta x=h$", (a+h_value/2, fa),
                        xytext=(0, -24), textcoords="offset points", ha="center")
            ax.annotate(r"$\Delta y$", (a+h_value, (fa+faq)/2),
                        xytext=(10, 0), textcoords="offset points", va="center")

            ax.axhline(0, linewidth=0.8)
            ax.axvline(0, linewidth=0.8)
            ax.grid(alpha=0.2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"a = {a:.3f}   |   h = {h_value:.5f}")
            adaptive_ylim(ax, yy, ysec, ytan)
            ax.legend(loc="best")
            return fig, m_sec, faq, delta_y

        placeholder = col_plot.empty()

        if animar:
            hs = np.geomspace(max(h, 0.5), 0.01, 30)
            last = None
            for hv in hs:
                fig, msec, faq, dy = make_derivative_plot(float(hv))
                placeholder.pyplot(fig, clear_figure=True)
                plt.close(fig)
                last = (float(hv), msec, faq, dy)
                time.sleep(0.08)
            h_used, m_sec, faq, delta_y = last
        else:
            fig, m_sec, faq, delta_y = make_derivative_plot(h)
            placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)
            h_used = h

        erro = abs(m_sec - deriv_a)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("h = Δx", f"{h_used:.6f}")
        m2.metric("Δy", f"{delta_y:.6f}")
        m3.metric("Inclinação secante", f"{m_sec:.8f}")
        m4.metric("f'(a)", f"{deriv_a:.8f}")

        st.latex(r"m_{\mathrm{sec}}=\frac{f(a+h)-f(a)}{h}")
        st.latex(r"f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}")

        st.write(f"Derivada simbólica:  **f'(x) = {sp.sstr(dexpr)}**")
        st.write(f"Erro entre a inclinação da secante e f'(a): **{erro:.3e}**")

    except Exception as e:
        st.error(f"Não foi possível processar a função: {e}")

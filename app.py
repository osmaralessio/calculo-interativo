
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Cálculo Iterativo",
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
    ["🏠 Início", "▥ Soma de Riemann", "📈 Derivada gráfica", "🔵 Método de Newton", "🔶 GeoGebra Book"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Aplicativos interativos para o ensino de Cálculo.")

# ---------- Início ----------
if pagina == "🏠 Início":
    st.title("Cálculo Interativo")
    st.subheader("Visualizações computacionais para o ensino de Cálculo")

    st.write(
        "Esta versão reúne três aplicativos interativos em Python: "
        "Soma de Riemann, Interpretação Gráfica da Derivada e Método de Newton."
    )

    c1, c2, c3, c4 = st.columns(4)
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
    with c3:
        st.info(
            "### 🔵 Método de Newton\n"
            "Acompanhe geometricamente as retas tangentes sucessivas e "
            "as aproximações de uma raiz de f(x)=0."
        )
    with c4:
        st.info(
            "### 🔶 GeoGebra Book\n"
            "Acesse a coleção de atividades interativas organizadas em capítulos "
            "no GeoGebra."
        )

    st.markdown("### Ideia pedagógica")
    st.write(
        "Os controles permitem que o estudante altere parâmetros e veja imediatamente "
        "como as construções geométricas e os valores numéricos se modificam."
    )

    st.markdown("### Próximos módulos possíveis")
    st.write(
        "Segunda derivada e concavidade, Teorema do Valor Médio, Bisseção, "
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
elif pagina == "📈 Derivada gráfica":
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



# ---------- GeoGebra Book ----------
elif pagina == "🔶 GeoGebra Book":
    st.title("🔶 GeoGebra Book")
    st.write(
        "Acesse o livro **Cálculo Interativo com GeoGebra**, com atividades "
        "organizadas em capítulos de Cálculo, Métodos Numéricos, Números Complexos "
        "e Geometria."
    )

    # Link público do Book. Se o seu endereço público for diferente,
    # basta trocar somente a linha abaixo.
    BOOK_URL = "https://www.geogebra.org/m/yw9snazg"

    st.link_button(
        "📘 Abrir o GeoGebra Book em uma nova aba",
        BOOK_URL,
        use_container_width=True
    )

    st.markdown("### Visualização dentro do site")

    # Alguns recursos do GeoGebra podem bloquear incorporação dependendo
    # das configurações do material. Se isso ocorrer, use o botão acima.
    components.html(
        f"""
        <iframe
            src="{BOOK_URL}"
            width="100%"
            height="850"
            style="border:1px solid #ddd; border-radius:10px;"
            allowfullscreen>
        </iframe>
        """,
        height=880,
        scrolling=True
    )

    st.caption(
        "Se o livro não aparecer incorporado, clique no botão acima para abri-lo "
        "diretamente no GeoGebra."
    )

# ---------- Método de Newton ----------
elif pagina == "🔵 Método de Newton":
    st.title("🔵 Método de Newton")
    st.write(
        "Encontre numericamente uma raiz de f(x)=0 usando retas tangentes sucessivas."
    )

    st.latex(r"x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}")

    col_controls, col_plot = st.columns([1, 2], gap="large")

    with col_controls:
        funcao_txt = st.text_input(
            "Função f(x)",
            "x**3 - x - 2",
            key="n_func"
        )

        x0 = st.number_input(
            "Valor inicial x₀",
            value=1.5,
            step=0.1,
            format="%.6f",
            key="n_x0"
        )

        tol = st.number_input(
            "Tolerância",
            value=1e-6,
            min_value=1e-12,
            max_value=1.0,
            format="%.1e",
            key="n_tol"
        )

        max_iter = st.slider(
            "Máximo de iterações",
            min_value=1,
            max_value=50,
            value=10,
            key="n_max_iter"
        )

        criterio = st.selectbox(
            "Critério de parada",
            [
                "|xₖ₊₁ - xₖ| < tolerância",
                "|f(xₖ₊₁)| < tolerância"
            ],
            key="n_criterio"
        )

    try:
        expr = parse_function(funcao_txt)
        dexpr = sp.diff(expr, x)

        f = numpy_function(expr)
        df = numpy_function(dexpr)

        dados = []
        xk = float(x0)
        convergiu = False
        motivo = ""

        for k in range(max_iter):
            fxk = float(f(xk))
            dfxk = float(df(xk))

            if not np.isfinite(fxk) or not np.isfinite(dfxk):
                motivo = "A função ou a derivada produziu valor não finito."
                break

            if abs(dfxk) < 1e-14:
                motivo = (
                    f"A derivada ficou muito próxima de zero em x = {xk:.10f}. "
                    "O método foi interrompido."
                )
                break

            xnext = xk - fxk / dfxk
            fxnext = float(f(xnext))
            erro = abs(xnext - xk)

            dados.append({
                "k": k,
                "x_k": xk,
                "fx_k": fxk,
                "dfx_k": dfxk,
                "x_next": xnext,
                "erro": erro,
                "fx_next_abs": abs(fxnext),
            })

            if criterio.startswith("|x"):
                atingiu = erro < tol
            else:
                atingiu = abs(fxnext) < tol

            xk = xnext

            if atingiu:
                convergiu = True
                motivo = f"Critério de parada atingido na iteração {k+1}."
                break

        if not dados:
            st.error(motivo if motivo else "Não foi possível iniciar as iterações.")
            st.stop()

        raiz = dados[-1]["x_next"]
        fx_raiz = float(f(raiz))

        iter_visual = st.slider(
            "Iteração mostrada no gráfico",
            min_value=0,
            max_value=len(dados)-1,
            value=len(dados)-1,
            key="n_iter_visual"
        )

        linha = dados[iter_visual]
        xkg = linha["x_k"]
        fxkg = linha["fx_k"]
        dfxkg = linha["dfx_k"]
        xnextg = linha["x_next"]

        xs = [d["x_k"] for d in dados] + [d["x_next"] for d in dados]
        xmin0 = min(xs)
        xmax0 = max(xs)
        span = max(2.0, (xmax0 - xmin0) * 0.8 + 1.0)

        xx = np.linspace(xmin0 - span, xmax0 + span, 1400)
        yy = finite_array(f(xx), xx)
        tang = fxkg + dfxkg * (xx - xkg)

        with col_plot:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(xx, yy, linewidth=3, label="f(x)")
            ax.plot(
                xx,
                tang,
                "--",
                linewidth=2.2,
                label=f"Tangente em x_{linha['k']} = {xkg:.6f}"
            )

            ax.axhline(0, linewidth=1)
            ax.axvline(0, linewidth=0.8)

            ax.scatter(
                [xkg],
                [fxkg],
                s=110,
                zorder=6,
                label=f"Pₖ = ({xkg:.5f}, {fxkg:.5f})"
            )

            ax.scatter(
                [xnextg],
                [0],
                s=110,
                zorder=6,
                label=f"xₖ₊₁ = {xnextg:.8f}"
            )

            ax.plot(
                [xkg, xkg],
                [0, fxkg],
                ":",
                linewidth=1.5
            )

            ax.annotate(
                f"xₖ={xkg:.5f}",
                (xkg, 0),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center"
            )

            ax.annotate(
                f"xₖ₊₁={xnextg:.5f}",
                (xnextg, 0),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center"
            )

            ax.grid(alpha=0.22)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(
                f"Método de Newton — iteração {linha['k']} → {linha['k']+1}"
            )

            adaptive_ylim(ax, yy, tang)
            ax.legend(loc="best")
            st.pyplot(fig, clear_figure=True)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Aproximação da raiz", f"{raiz:.10f}")
        m2.metric("f(raiz)", f"{fx_raiz:.3e}")
        m3.metric("Iterações", len(dados))
        m4.metric("Erro final", f"{dados[-1]['erro']:.3e}")

        st.markdown("### Derivada")
        st.latex(r"f'(x)=" + sp.latex(dexpr))

        if convergiu:
            st.success(motivo)
        else:
            st.warning(
                motivo if motivo
                else "O número máximo de iterações foi atingido sem satisfazer o critério."
            )

        st.markdown("### Tabela de iterações")

        tabela = []
        for d in dados:
            tabela.append({
                "k": d["k"],
                "x_k": f"{d['x_k']:.10f}",
                "f(x_k)": f"{d['fx_k']:.6e}",
                "f'(x_k)": f"{d['dfx_k']:.6e}",
                "x_k+1": f"{d['x_next']:.10f}",
                "|x_k+1 - x_k|": f"{d['erro']:.6e}",
                "|f(x_k+1)|": f"{d['fx_next_abs']:.6e}",
            })

        st.dataframe(tabela, use_container_width=True)

        st.markdown("### Interpretação geométrica")
        st.write(
            "Em cada passo, traçamos a reta tangente ao gráfico de f em "
            "Pₖ=(xₖ,f(xₖ)). A interseção dessa tangente com o eixo x fornece "
            "a próxima aproximação xₖ₊₁."
        )

        st.latex(r"y-f(x_k)=f'(x_k)(x-x_k)")
        st.write("Fazendo y=0 na equação da tangente, obtemos:")
        st.latex(r"x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}")

    except Exception as e:
        st.error(f"Não foi possível executar o Método de Newton: {e}")

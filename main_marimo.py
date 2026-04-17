import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    from Functions.implement import rate,getThetas,getXrc

    return getThetas, getXrc, mo, np, pd, plt, rate


@app.cell
def _(mo, np):
    sliders = mo.ui.dictionary({
    "f": mo.ui.slider(steps=np.linspace(0, 30, 101), value=0, label="F", show_value=True),
                    "BDFE": mo.ui.slider(steps=np.linspace(0, 3, 101), value=1., label="BDFE", show_value=True),
                    "log kva": mo.ui.slider(steps=np.linspace(-15, 15, 301), value=-1, label="Log kva", show_value=True),
                    "l va": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="L va", show_value=True),
                    "b va": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="B va", show_value=True),
                    "log kvb": mo.ui.slider(steps=np.linspace(-15, 15, 301), value=-1, label="Log kvb", show_value=True),
                    "l vb": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="L vb", show_value=True),
                    "b vb": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="B vb", show_value=True),
                    "log kt": mo.ui.slider(steps=np.linspace(-15, 15, 301), value=-1, label="Log kt", show_value=True),
                    "l t": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="L t", show_value=True),
                    "log kha": mo.ui.slider(steps=np.linspace(-15, 15, 301), value=-1, label="Log kha", show_value=True),
                    "l ha": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="L ha", show_value=True),
                    "b ha": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="B ha", show_value=True),
                    "log khb": mo.ui.slider(steps=np.linspace(-15, 15, 301), value=-1, label="Log khb", show_value=True),
                    "l hb": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="L hb", show_value=True),
                    "b hb": mo.ui.slider(steps=np.linspace(0, 1, 101), value=0.5, label="B hb", show_value=True),

                    "E_range": mo.ui.range_slider(steps=np.linspace(-5, 0.5, 551), value=(-1.5, 0), label="E range", show_value=True),
                    "E_select": mo.ui.slider(steps=np.linspace(-5, 0.5, 551), value=-1.5, label="E select", show_value=True),

                    "pH": mo.ui.slider(steps=np.linspace(0, 14, 141), value=7, label="pH", show_value=True),

                # "T": mo.ui.slider(start=273, stop=373,value=298,label="Temperature (K)"),
            })


    slider_stack = mo.vstack([val for val in sliders.values()])
    return slider_stack, sliders


@app.cell
def _(getThetas, getXrc, np, pd, plt, rate, sliders):
    #f,BDFE,log k1f, l1,b1,log k2f,l2,b2, log kt,lt,log kh1f, lh1,bh1,log kh2f,lh2,bh2
    params = [sliders["f"].value,
                    sliders["BDFE"].value,
                    sliders["log kva"].value,
                    sliders["l va"].value,
                    sliders["b va"].value,
                    sliders["log kvb"].value,
                    sliders["l vb"].value,
                    sliders["b vb"].value,
                    sliders["log kt"].value,
                    sliders["l t"].value,
                    sliders["log kha"].value,
                    sliders["l ha"].value,
                    sliders["b ha"].value,
                    sliders["log khb"].value,
                    sliders["l hb"].value,
                    sliders["b hb"].value]

    Es = np.linspace(sliders["E_range"].value[0],sliders["E_range"].value[1],101)
    pH = sliders["pH"].value
    Xs = np.array([Es, [pH]*len(Es)])
    thetas = getThetas(params, Xs,debug=False)
    rates = rate(params, Xs,debug=False,thetas=thetas)
    Xrcs = getXrc(params, *Xs,dlogk=0.001)


    mask = rates != -100.0
    Xrcs = pd.DataFrame(Xrcs, index=Es, columns=['Xva', 'Xvb', 'Xt', 'Xha', 'Xhb'])
    Xrcs.drop(Xrcs.index[~mask], inplace=True)
    clipped = Xrcs.clip(lower=0, upper=1)
    clipped.plot.area(stacked=True, figsize=(10, 6))
    plt.plot(Es[mask], thetas[mask], color='cyan', label='theta')
    plt.xlabel('E vs SHE (V)')
    plt.ylabel('Rate control (Xrc)')
    plt.legend()
    ax2 = plt.twinx()
    ax2.plot(Es[mask], rates[mask], color='black', label='log(rate)')
    plt.ylabel('log(rate)')
    fig = plt.gcf()
    return (fig,)


@app.cell
def _(fig, mo, slider_stack):
    mo.hstack([slider_stack,fig])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

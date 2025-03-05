# from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from tqdm.auto import tqdm
from IPython import get_ipython


def df_columns():
    colx = [
        "b_x",
        "l1_x",
        "l2_x",
        "l3_x",
        "l4_x",
        "l5_x",
        "l6_x",
        "l7_x",
        "l8_x",
        "l9_x",
        "l10_x",
        "l11_x",
        "r1_x",
        "r2_x",
        "r3_x",
        "r4_x",
        "r5_x",
        "r6_x",
        "r7_x",
        "r8_x",
        "r9_x",
        "r10_x",
        "r11_x",
    ]
    coly = [
        "b_y",
        "l1_y",
        "l2_y",
        "l3_y",
        "l4_y",
        "l5_y",
        "l6_y",
        "l7_y",
        "l8_y",
        "l9_y",
        "l10_y",
        "l11_y",
        "r1_y",
        "r2_y",
        "r3_y",
        "r4_y",
        "r5_y",
        "r6_y",
        "r7_y",
        "r8_y",
        "r9_y",
        "r10_y",
        "r11_y",
    ]

    colors = [
        "b",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
    ]
    return colx, coly, colors


def np_column():
    colx = [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
    ]
    coly = [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        33,
        35,
        37,
        39,
        41,
        43,
        45,
        47,
    ]
    colors = [
        "b",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "r",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
        "g",
    ]
    return colx, coly, colors


def f_df(df, ax, colx, coly, colors, marker="o", line_style="-", alpha=1) -> list:
    ims = []
    for i in tqdm(range(len(df)), leave=True):
        im_ls = []
        for x, y, c in zip(colx, coly, colors):
            (im,) = ax.plot(
                df[x][:i],
                df[y][:i],
                c,
                label=x,
                linestyle=line_style,
                alpha=alpha,
                linewidth=0.5,
            )
            im_ls.append(im)
            im = ax.scatter(df[x][i], df[y][i], color=c, marker=marker, alpha=alpha)
            im_ls.append(im)
        ims.append(im_ls)
    return ims


def f_np(ndarray, ax, colx, coly, colors, marker="o", line_style="-", alpha=1) -> list:
    ims = []
    for i in tqdm(range(ndarray.shape[0]), leave=True):
        im_ls = []
        for x, y, c in zip(colx, coly, colors):
            (im,) = ax.plot(
                ndarray[: i + 1, x],
                ndarray[: i + 1, y],
                c,
                label=x,
                linestyle=line_style,
                alpha=alpha,
                linewidth=0.5,
            )
            im_ls.append(im)
            im = ax.scatter(
                ndarray[i, x], ndarray[i, y], color=c, marker=marker, alpha=alpha
            )
            im_ls.append(im)
        ims.append(im_ls)
    return ims


def visualize(df):
    colx, coly, colors = df_columns()
    fig, ax = plt.subplots()
    df = df.tail(50)
    l_name = df["l_name"].unique()[0]
    r_name = df["r_name"].unique()[0]
    ax.set_title(f"{l_name} vs {r_name}")
    ims = f_df(df, ax, colx, coly, colors)
    anim = animation.ArtistAnimation(fig, ims, interval=100)
    if get_ipython() is not None:
        rc("animation", html="jshtml")
    return anim


def visualizer_np(ndarray):
    colx, coly, colors = np_column()
    fig, ax = plt.subplots()
    l_name = ndarray[0][0]
    r_name = ndarray[0][1]
    ax.set_title(f"{l_name} vs {r_name}")
    ims = f_np(ndarray, ax, colx, coly, colors)
    anim = animation.ArtistAnimation(fig, ims, interval=100)

    if get_ipython() is not None:
        rc("animation", html="jshtml")
    return anim


def visualizer_df2(df1, df2):
    colx, coly, colors = df_columns()
    fig, ax = plt.subplots()
    df1 = df1.tail(50)
    df2 = df2.tail(50)
    l_name = df1["l_name"].unique()[0]
    r_name = df1["r_name"].unique()[0]
    ax.set_title(f"{l_name} vs {r_name}")
    ims1 = f_df(df1, ax, colx, coly, colors, marker="o", line_style="-")
    ims2 = f_df(df2, ax, colx, coly, colors, marker="x", line_style="--")
    ims = [[*a, *b] for a, b in zip(ims1, ims2)]
    anim = animation.ArtistAnimation(fig, ims, interval=100)
    if get_ipython() is not None:
        rc("animation", html="jshtml")
    return anim


def visualizer_np2(ndarray1, ndarray2):
    colx, coly, colors = np_column()
    fig, ax = plt.subplots()
    ims = []
    l_name = ndarray1[0][0]
    r_name = ndarray1[0][1]
    ax.set_title(f"{l_name} vs {r_name}")
    ims1 = f_np(ndarray1, ax, colx, coly, colors, marker="o", line_style="-")
    ims2 = f_np(ndarray2, ax, colx, coly, colors, marker="x", line_style="--")
    ims = [[*a, *b] for a, b in zip(ims1, ims2)]
    anim = animation.ArtistAnimation(fig, ims, interval=100)
    if get_ipython() is not None:
        rc("animation", html="jshtml")
    plt.close()
    return anim


def plot(df):
    colx, coly, colors = df_columns()
    fig, ax = plt.subplots()
    df = df.tail(50)
    ims = f_df(df, ax, colx, coly, colors)
    anim = animation.ArtistAnimation(fig, ims, interval=100)
    if get_ipython() is not None:
        rc("animation", html="jshtml")
    return anim


def col2index(cols):
    d = {}

    for i, col in enumerate(cols):
        if col.endswith("_x"):
            if col[:-2] in d:
                d[col[:-2]][0] = i
            else:
                d[col[:-2]] = [i, None]
        if col.endswith("_y"):
            if col[:-2] in d:
                d[col[:-2]][1] = i
            else:
                d[col[:-2]] = [None, i]
    colx = [d[i][0] for i in d]
    coly = [d[i][1] for i in d]
    colors = []
    for key, value in d.items():
        if key.startswith("b"):
            colors.append("b")
        elif key.startswith("l"):
            colors.append("r")
        elif key.startswith("r"):
            colors.append("g")
    return colx, coly, colors


def plot_np2(ndarray1, ndarray2, cols: list[str]):
    colx, coly, colors = col2index(cols)
    fig, ax = plt.subplots()
    ims = []
    ims1 = f_np(ndarray1, ax, colx, coly, colors, marker="o", line_style="-")
    ims2 = f_np(ndarray2, ax, colx, coly, colors, marker="x", line_style="--")
    ims = [[*a, *b] for a, b in zip(ims1, ims2)]
    anim = animation.ArtistAnimation(fig, ims, interval=100)
    if get_ipython() is not None:
        rc("animation", html="jshtml")
    plt.close()
    return anim

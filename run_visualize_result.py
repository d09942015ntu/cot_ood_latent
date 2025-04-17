from collections import defaultdict
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def get_vals(dname):
    vals_1 = defaultdict(list)
    vals_2 = defaultdict(list)
    for s in range(2):

        log_file = open(f"outputs/{dname}_lrelu_l2_s{str(s).zfill(2)}/trainer.log", "r")
        log_lines = log_file.readlines()
        log_line_json = json.loads(log_lines[-1])

        for ikey, ival in log_line_json['train_loss'].items():
            if '_1' in ikey:
                vals_1[ikey].append(log_line_json['train_loss'][ikey])
            if '_2' in ikey:
                vals_2[ikey].append(log_line_json['train_loss'][ikey])
        for ikey, ival in log_line_json['test_loss'].items():
            if '_1' in ikey:
                vals_1[ikey].append(log_line_json['test_loss'][ikey])
            if '_2' in ikey:
                vals_2[ikey].append(log_line_json['test_loss'][ikey])

    return vals_1, vals_2


def draw_plot(point_arrays, output_name):
    plt.clf()
    for pname, point_array in point_arrays.items():
        X = [x[0] for x in point_array]
        Y = [x[1] for x in point_array]
        print(f"{pname}, {X}, {Y}")
        plt.plot(X, Y, label=pname)
    plt.legend()
    plt.ylim(0.00001, 1.1)
    plt.xlabel("$KL(D_{train}|D_{testA})$")
    plt.ylabel("$\mathcal{L}_{\mathrm{test}}(h)$")
    plt.yscale("log")
    plt.savefig(os.path.join(f"imgs", f"{output_name}.png"), bbox_inches='tight')


def draw_tikz(point_arrays, output_name, text_title="", text_xlabel="", x_label_x_shift=0):
    text_ylabel = "$\mathcal{L}_{\mathrm{test}}(h)$"
    configs = ("""
    \\begin{tikzpicture}
    \\begin{axis}[
        width=8cm,
        height=5cm,
        legend pos=outer north east,
        grid=major,
        grid style={dashed,gray!30},
        xlabel={TEXT_XLABEL},
        ylabel={TEXT_YLABEL},
        title={TEXT_TITLE},
        ymin=0.0001,
        ymax=1,
        ymode=log,
        font=\\scriptsize,
        xlabel style={
            at={(current axis.south east)}, % Relative positioning
            anchor=north east,              % Anchoring at a specific point
            yshift=15pt,                   % Shifting downward
            xshift=X_LABEL_X_SHIFTpt                      % Shifting rightward (if needed)
        },
        ylabel style={
            at={(current axis.north west)}, % Relative positioning
            anchor=north east,              % Anchoring at a specific point
            yshift=-10pt,                   % Shifting downward
            xshift=10pt                      % Shifting rightward (if needed)
        },
        legend style={
            font=\\scriptsize
        },
        title style={
            font=\\normal
        }
    ]
    """
               .replace("TEXT_XLABEL", text_xlabel)
               .replace("TEXT_YLABEL", text_ylabel)
               .replace("TEXT_TITLE", text_title)
               .replace("X_LABEL_X_SHIFT", str(x_label_x_shift))
               )
    out_lines = []
    min_y = 999
    for i, (pname, point_array) in enumerate(sorted(point_arrays.items(), key=lambda x: x[0], reverse=True)):
        X = [x[0] for x in point_array]
        Y = [x[1] for x in point_array]
        min_y = min(min_y, min(Y))
        if "train" in pname:
            out_lines.append("\\addplot[pc%s, thick] table[row sep=\\\\]{\nx y \\\\ \n" % (i + 1))
        else:
            out_lines.append("\\addplot[pc%s, thick, dashed] table[row sep=\\\\]{\nx y \\\\ \n" % (i + 1))
        for x, y in zip(X, Y):
            out_lines.append(f" {x}  {y} \\\\ \n")
        out_lines.append("};\n")
        pname_label = (pname.replace("Z", " ~\\Theta")
                       .replace("B", " ~\\widetilde{\\Theta}")
                       .replace("C", " ~\\bar{\\Theta}")
                       .replace("_1", ",~h=1")
                       .replace("_2", ",~h=2")
                       .replace("train", "\\text{Training Set}")
                       .replace("test", "\\text{Testing Set}")
                       )
        out_lines.append("\\addlegendentry{$%s$}\n" % pname_label)
    out_lines.append("""\\end{axis}\n \\end{tikzpicture}\n """)
    configs = configs.replace("MIN_Y", str(min_y * 0.1))
    f = open(os.path.join(f"imgs", f"{output_name}.tex"), "w")
    f.write(configs)
    f.write("".join(out_lines))


def run_tilde():
    vals_data = defaultdict(list)
    for prob in [5, 10, 20, 30, 40, 50]:
        dname = f"discrete14_3_{str(prob).zfill(2)}"
        vals_1, vals_2 = get_vals(dname)
        for ikey, ivals in vals_1.items():
            if "testC" not in ikey:
                vals_data[ikey].append((0.01 * prob / (1 - 0.01 * prob), np.average(ivals)))
        for ikey, ivals in vals_2.items():
            if "testC" not in ikey:
                vals_data[ikey].append((0.01 * prob / (1 - 0.01 * prob), np.average(ivals)))

    draw_plot(vals_data, f"loss_tilde")
    draw_tikz(vals_data, f"loss_tilde", text_xlabel="$|\\tilde{\\Theta}|/|\\Theta|$", x_label_x_shift=20)


def run_bar():
    vals_data = defaultdict(list)
    for prob in [5, 10, 20, 30, 40, 50]:
        dname = f"discrete14_3_{str(prob).zfill(2)}"
        vals_1, vals_2 = get_vals(dname)
        for ikey, ivals in vals_1.items():
            if "testB" not in ikey:
                vals_data[ikey].append(np.average(ivals))
        for ikey, ivals in vals_2.items():
            if "testB" not in ikey:
                vals_data[ikey].append(np.average(ivals))
    vals_data2 = {}
    for ikey, ivals in vals_data.items():
        vals_data2[ikey] = np.average(ivals)

    k11 = 'testC, ~p=1-\delta, ~h=1'
    k12 = 'testC, ~p=1-\delta, ~h=2'
    k21 = 'testC, ~p=1+\delta, ~h=1'
    k22 = 'testC, ~p=1+\delta, ~h=2'
    vals_data3 = defaultdict(list)
    for delta in [5, 10, 20, 30, 40, 50]:
        vals_data3[k11].append((delta / 100, vals_data2[f'testC{str(100 - delta).zfill(3)}_1']))
        vals_data3[k12].append((delta / 100, vals_data2[f'testC{str(100 - delta).zfill(3)}_2']))
        vals_data3[k21].append((delta / 100, vals_data2[f'testC{100 + delta}_1']))
        vals_data3[k22].append((delta / 100, vals_data2[f'testC{100 + delta}_2']))
        vals_data3['train_1'].append((delta / 100, vals_data2['train_1']))
        vals_data3['train_2'].append((delta / 100, vals_data2['train_2']))
        vals_data3['testZ_1'].append((delta / 100, vals_data2['testZ_1']))
        vals_data3['testZ_2'].append((delta / 100, vals_data2['testZ_2']))

    draw_plot(vals_data3, f"loss_bar")
    draw_tikz(vals_data3, f"loss_bar", text_xlabel="$\\delta$", x_label_x_shift=10)


if __name__ == '__main__':
    run_tilde()
    run_bar()

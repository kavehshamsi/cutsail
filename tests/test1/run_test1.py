import sys
import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np

kvals = [2, 3, 4, 5, 6, 7]
sgvals = [3, 6, 8, 18]
cat_targets = ['', '-catT']
g_methods = ['tt', 'nauty', 'adjmat', 'adjlist']
g_method_options = ['', '-indsoft', '-indsoft', '-indsoft']
ml_methods = ['nn', 'gcn', 'ruledict']
ml_method_options = ['-bs=50 -pc=8 -nep=250', '-bs=50 -pc=8 -nep=250', '']

bench_glob = '../../bench/iscas_abc_simple/c*'
log_dir = './logs/'
run_script = '../../oless_lut_master.py'

num_tests = len(kvals) * len(g_methods) * len(ml_methods) * len(sgvals)

def worker(cmd):
    print('----runing----', cmd)
    os.system(cmd)
    print('----done with---- ', cmd)

def analyze_log(log_file):
    acc = -1
    nacc = None
    with open(log_file, 'r') as fn:
        for ln in fn:
            if 'perfect match' in ln:
                #print(ln)
                x = ln.split('->')[-1][0:-2]
                #print(x)
                acc = float(x)
            if 'naive match' in ln:
                x = ln.split('->')[-1][0:-2]
                nacc = float(x)
    return acc, nacc


pool = mp.Pool(4)

if input('do you want to run {} tests? [yes/no]'.format(num_tests)) == 'yes':
    print('running tests')

    for sgval in sgvals:
        for kval in kvals:
            for gmind, g_method in enumerate(g_methods):
                for mlmind, ml_method in enumerate(ml_methods):
                    for cat_target in cat_targets:
          
                        cmd = f'python3.9 -u {run_script} -tE={g_method} {g_method_options[gmind]} {cat_target} -K={kval} -sG={sgval} -mlm={ml_method} {ml_method_options[mlmind]} \'{bench_glob}\''
                        log_file = log_dir + 'log_k{}_sg{}_mlm{}_cT{}_gm{}.txt'.format(kval, sgval, cat_target.replace('-',''), ml_method, g_method)
          
                        if cat_target == '-catT' and ml_method == 'ruledict':
                            #os.system(f'rm {log_file}')
                            continue
                        
                        cmd += f' | tee {log_file} '
                        # assert worker(cmd) == 0
                        if not os.path.exists(log_file) or analyze_log(log_file)[0] == -1:
                            #print(cmd)
                            pool.apply_async(worker, args=(cmd,))

    pool.close()
    pool.join()
    print('FINISHED!')

    
gen_data = True
if gen_data:

    # kvals # g_method # ml_method # data
    dat = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

    for sgval in sgvals:
        for kval in kvals:
            for gmind, g_method in enumerate(g_methods):
                for mlmind, ml_method in enumerate(ml_methods):
                    for cat_target in cat_targets:
                    
                        log_file = log_dir + 'log_k{}_sg{}_mlm{}_cT{}_gm{}.txt'.format(kval, sgval, cat_target.replace('-',''), ml_method, g_method)
            
                        if cat_target == '-catT' and ml_method == 'ruledict':
                            # os.system(f'rm {log_file}')
                            continue
                            
                        acc = analyze_log(log_file)
                        dat[sgval][kval][g_method][ml_method][cat_target] = acc
                        
          
    plot_data = False
    if plot_data:
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            'axes.axisbelow': True
        })

        plt.rcParams.update( {'figure.figsize' : (15, 5)} )
        fig, axs = plt.subplots(nrows=len(ml_methods), ncols=len(sgvals), sharey=True, sharex=True, figsize=(14, 3.2))
        plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, hspace=0.2, wspace=0.14)
        plt.ylim(0, 100)

        width = 0.15
        step = width * 4
        colors = ['red', 'blue', 'black', 'gray']
        #markers = ['v', 'o', '^']
        poses = [-3*width/2, -width/2, width/2, 3*width/2]

        ml_methods = ['gcn']
        
        for mlind, ml_method in enumerate(ml_methods):    
            for sgind, sgval in enumerate(sgvals):
                ax = plt.subplot(len(ml_methods), len(sgvals), (mlind * len(sgvals)) + sgind+1)
                Y = [list() for i in range(len(g_methods))]
                for gind, g_method in enumerate(g_methods):
                    for kval in kvals:
                        #Y[gind].append(dat[sgval][kval][g_method][ml_method][''])
    #                    if kval > 4:
    #                        Y[gind].append(dat[sgval][kval][g_method][ml_method][''])
    #                    else:
                        Y[gind].append(dat[sgval][kval][g_method][ml_method]['-catT'])
                    print(Y[gind])
                    plt.bar(np.array(kvals) + poses[gind],
                            Y[gind], color=colors[gind], label=g_methods[gind], edgecolor='black', width=width)#, alpha=0.7)

                plt.xticks(kvals, fontsize=13)
                plt.title('l={}'.format(sgval), fontsize=15, y=1)
                plt.grid(linestyle='--')
                plt.ylim(0, 100)

                #plt.yticks([0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                
                if sgind == 0:
                    plt.legend(fontsize=14, ncol=1) #, loc=(0, -0.4))
                    plt.ylabel("accuracy", fontsize=16)
                    
                plt.xlabel("K", fontsize=16)



        plt.savefig('./figs/accbar.pdf')
        #plt.show()
    
    kvals = [2, 3, 4, 5, 6, 7]
    ml_methods = ['ruledict', 'nn', 'gcn']
    
    g_method = 'tt'
    print('\n\n\n \\begin{tabular}{C{0.25in}|C{0.25in}|C{0.22in}', end='')
    for sgval in sgvals:
        for i in range(len(ml_methods)-1):
            print('|C{{{}in}}'.format(0.22, 0.22), end='')
    print('} \\hline \\hline')
    
    for sgval in sgvals:
        print(' & \multicolumn{{{}}}{{|c}}{{sG={}}} '.format(len(ml_methods), sgval), end='')
    print('\\\\ \\hline')
    
    
    print('K & catY ', end='')
    for sgind, sgval in enumerate(sgvals):
        for ml_method in ml_methods:
            if ml_method == 'ruledict':
                if sgind == 0:
                    print('& nv ', end='')
                else:
                    pass
            else: 
                print('& {} '.format(ml_method.upper()), end='')
    print('\\\\ \\hline')
    
    for kval in kvals:
        for cat_target in cat_targets:
            if cat_target == '':
                print('\\multirow{{2}}*{{{}}} & - '.format(kval), end='')
            else:
                print('  & \\checkmark '.format(), end='')
            for sgind, sgval in enumerate(sgvals):
                for ml_method in ml_methods:

                    try:
                        if ml_method == 'ruledict':
                            if sgind == 0:
                                print( ' & \multirow{{2}}*{{{:.3g}}}'.format( dat[sgval][kval][g_method][ml_method][cat_target][1] ), end='')
                        else:
                            print( ' & {:.3g}'.format( dat[sgval][kval][g_method][ml_method][cat_target][0] ), end='')
                    except KeyError:
                        print( ' &  ', end='')
            print('\\\\')
        print(' \\hline')

    print('\\hline \\hline \n \\end{tabular} \n\n\n\n ')
   


import sys

import multiprocessing as mp
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np

kvals = [2, 3, 4, 5, 6, 7]
sgvals = [3, 5, 6, 8, 12, 18]
g_methods = ['tt'] #, 'nauty', 'adjmat', 'adjlist']
g_method_options = ['', '-indsoft', '-indsoft', '-indsoft']
ml_methods = ['ruledict']
ml_method_options = ['']

bench_glob = './sim/*'
log_dir = './logs/'
run_script = '../../oless_lut_master.py'

num_tests = len(kvals) * len(g_methods) * len(ml_methods) * len(sgvals)

def worker(cmd):
    print(cmd)
    os.system(cmd)
    print('done with ', cmd)

def analyze_log(log_file):
    acc = -1
    with open(log_file, 'r') as fn:
        for ln in fn:
            if 'perfect match' in ln:
                print(ln)
                x = ln.split('->')[-1][0:-2]
                print(x)
                acc = float(x)
    return acc


pool = mp.Pool(40)

if input('do you want to run {} tests? [yes/no]'.format(num_tests)) == 'yes':
    print('running tests')

    for sgval in sgvals:
        for kval in kvals:
            for gmind, g_method in enumerate(g_methods):
                for mlmind, ml_method in enumerate(ml_methods):
                    cat_target = ''
                    cmd = f'python3.9 -u {run_script} -tE={g_method} {g_method_options[gmind]} -K={kval} -sG={sgval} -mlm={ml_method} {ml_method_options[mlmind]} \'{bench_glob}\''
                    log_file = log_dir + 'log_k{}_sg{}_mlm{}_cT{}_gm{}.txt'.format(kval, sgval, ml_method, cat_target.replace('-',''), g_method)
                                        
                    cmd += f' | tee {log_file} '
                    # assert worker(cmd) == 0
                    if not os.path.exists(log_file) or analyze_log(log_file) == -1:
                        # print(cmd)
                        pool.apply_async(worker, args=(cmd,))

    pool.close()
    pool.join()
    print('FINISHED!')


gen_data = True
if gen_data:

    # kvals # g_method # ml_method # data
    dat = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: defaultdict())))

    def analyze_log(log_file):
        acc = 0
        with open(log_file, 'r') as fn:
            for ln in fn:
                if 'num Xs ' in ln:
                    #print(ln)
                    x = ln.split(' ')[-1]
                    #print(x)
                    num_X = int(x)
                if 'num Ys ' in ln:
                    #print(ln)
                    x = ln.split(' ')[-1]
                    #print(x)
                    num_Y = int(x)
                if 'naive match' in ln:
                    nacc = float(ln.rstrip().split('->')[-1].replace('%', ''))
                    print(ln)
                if 'perfect match' in ln:
                    pacc = float(ln.rstrip().split('->')[-1].replace('%', ''))
        return num_X, num_Y, nacc, pacc

    for sgval in sgvals:
        for kval in kvals:
            for g_method in g_methods:
                for ml_method in ['ruledict']:
                    cat_target = ''
                    log_file = log_dir + 'log_k{}_sg{}_mlm{}_cT{}_gm{}.txt'.format(kval, sgval, ml_method, cat_target.replace('-',''), g_method)
                    dat[sgval][kval][g_method][ml_method] = analyze_log(log_file)
                    
    
    for sgval in sgvals:
        for kval in kvals:
            print('l={}, k={} acc={}'.format(sgval, kval, dat[sgval][kval]['tt']['ruledict'][2]))
                
    
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        'axes.axisbelow': True
    })

    plt.rcParams.update( {'figure.figsize' : (5, 4)} )
    plt.subplots_adjust(left=0.145, right=0.99, top=0.90, bottom=0.143, hspace=0.2, wspace=0.08)
    #plt.ylim(0, 100)

    width = 0.20
    step = width * 3
    colors = ['red', 'blue', 'black', 'gray', 'darkgreen', 'purple']
    markers = ['v', 'o', '^', '*', '.', '2']
    #poses = [-3*width/2, -width/2, width/2, 3*width/2]

    
    for sgind, sgval in enumerate([6]):
        for gind, g_method in enumerate(g_methods):

            xs = []
            for kval in kvals:
                d = dat[sgval][kval][g_method]['ruledict']
                print( 'kval={}, gm={}, x={} & y={} & '.format(kval, g_method, *d))
                xs.append(d[1])
            
            plt.plot(kvals, xs, c=colors[gind], marker=markers[gind], label=g_method)
            plt.grid()
            #plt.title('$|Y|$', fontsize=16)
            
            plt.xlabel("K", fontsize=16)
            plt.ylabel("$|CE|$", fontsize=16)
            plt.xticks(kvals)

    print()
    plt.grid(linestyle='--')
    plt.yscale('log')
    plt.legend(fontsize=16) #, loc=(0, -0.4))
    plt.savefig('./figs/catYs.pdf')
    plt.show()
    plt.cla()
    plt.clf()

    plt.subplots_adjust(left=0.148, right=0.99, top=0.90, bottom=0.16, hspace=0.2, wspace=0.08)
    for kind, kval in enumerate(kvals):
        xs = []
        for sgind, sgval in enumerate(sgvals):
            d = dat[sgval][kval]['tt']['ruledict']
            print( 'x={} & y={} & '.format(*d) , end='')
            xs.append(d[0])
            #plt.title('$|Y|$', fontsize=16)
        plt.plot(sgvals, xs, c=colors[kind], marker=markers[kind], label='k={}'.format(kval))
        plt.grid()
        plt.xlabel("$l$", fontsize=16)
        plt.ylabel("$|L_l|$", fontsize=16)
        plt.xticks(sgvals)
    
    print()
    plt.grid(linestyle='--')
    plt.yscale('log')
    plt.legend(fontsize=16) #, loc=(0, -0.4))
    plt.savefig('./figs/catXs.pdf')
    plt.show()


    # for ml_method in g_methods:
    #     print('{} & '.format(g_method))
    # for kval in kvals:
    #     print('k={} &'.format(kval))
    #         for g_method in g_methods:
    #             print(' {} & '.format(dat[kval][]))

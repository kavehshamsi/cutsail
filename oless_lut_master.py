import tensorflow.keras

import os, sys
from PIL.ImageOps import equalize

import circuit
import vparse
import liberty
import ccuts
import gencs

import pynauty

import numpy as np
import scipy.sparse as sp

import pickle, re
import time, datetime, resource, argparse, os, logging, psutil, time, copy, random, glob

from collections import defaultdict

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
#   except RuntimeError as e:
#     print(e)

import tensorflow_addons as tfa
import tensorflow.experimental.numpy as tnp

import sklearn
import sklearn.ensemble
import sklearn.neural_network
from sklearn.preprocessing import StandardScaler


import spektral
from spektral.data import Dataset, Graph
from spektral.models import GeneralGNN
from spektral.data.loaders import SingleLoader, DisjointLoader
from spektral.layers import GCNConv, GlobalSumPool
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms import GCNFilter

from spektral.data import Graph, Dataset, Loader, utils

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow import math

from sklearn.model_selection import train_test_split


def limit_memory(maxsize): 
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    print(soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard)) 

import matplotlib.pyplot as plt


def resynthesize(cir):
    tmpdir = myabc.create_temp_dir()
    presyn = tmpdir + 'presyn.bench'
    postsyn = tmpdir + 'postsyn.bench'
    cir.write_bench(presyn)
    myabc.resynthesize(presyn, postsyn, myabc.abclib, True)
    retcir = circuit.Circuit(postsyn)
    os.system('rm -rf {}'.format(tmpdir))
    return retcir

def pad_2dlist_to_maxlen(X, max_len=None):
    R = []
    if max_len is None:
        max_len = 0
        for i in range(len(X)):
            max_len = max(max_len, len(X[i]))
    for i in range(len(X)):
        R.append(resize_fill_list(X[i], max_len))
    return R

def categorize_2d(X):
    Xmap = dict()
    Xcat = []
    for xi in X:
        xsig = tuple(xi)
        if xsig not in Xmap:
            Xmap[xsig] = len(Xmap)
        Xcat.append([Xmap[xsig]])

    print('size categorization', len(Xmap))
    return Xcat, Xmap

def resize_fill_list(ls, sz, v=0):
    retL = copy.deepcopy(ls)
    if len(ls) < sz:
        retL.extend([v]*(sz - len(ls)))
    else:
        retL =  ls[:sz]
    return retL


def binary_encode(val, width):
    ret = []
    for i in reversed(range(width)):
        ret.append((val >> i) & 1)
    return ret


def one_hot_decode_np_exdim(X, num_cats):
    X_dec = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_dec[i, j] = np.argmax(X[i, j, 0:num_cats-1])
    return X_dec


def one_hot_decode_np_flat(X, num_cats):
    num_vals = int(X.shape[1] / num_cats)
    X_dec = np.zeros((X.shape[0], num_vals))
    for i in range(X.shape[0]):
        for j in range(num_vals):
            X_dec[i, j] = np.argmax(X[i, j * num_cats:(j + 1) * num_cats])
    return X_dec


def one_hot_encode_np_exdim0(X):
    cats = defaultdict(lambda: set())
    for i in range(len(X)):
        for j in range(len(X[i])):
            cats[j].add(X[i][j])

    cats = [list(y) for x, y in cats.items()]
    print(cats)

    eX = []
    for i in range(len(X)):
        eX.append([])
        for j in range(len(X[i])):
            eX[i].append([])
            eX[i][j] = one_hot_encode(X[i][j], len(cats[j]))
    return eX, cats


def one_hot_encode_np_exdim(X):
    num_cats = int(np.max(X)) + 1
    eX = np.zeros((X.shape[0], X.shape[1], num_cats))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            eX[i, j, :] = one_hot_encode(X[i, j], num_cats)
    return eX, num_cats


def one_hot_encode_np_flat(X):
    num_cats = int(np.max(X)) + 1
    eX = np.zeros((X.shape[0], X.shape[1] * num_cats))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            eX[i, j*num_cats:(j+1)*num_cats] = one_hot_encode(X[i, j], num_cats)
    return eX, num_cats


def one_hot_encode_mat(X):
    ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
    ohe.fit(X)
    return ohe.transform(X), ohe

def one_hot_encode(val, width):
    ret = [0 for i in range(width)]
    if val < len(ret):
        ret[val] = 1
    return ret


def one_hot_decode(ls):
    try:
        return(np.where(ls == 1)[0][0])
    except IndexError:
        return 0


def binary_to_int(ls, width, dir=0):
    ret = 0
    hi = max(len(ls), width)
    lo = 0
    if dir:
        hi, lo = lo, hi
        
    for i in range(lo, hi):
        ret = (ret << 1) | int(ls[i] == 1) 
        
    # print(ls[0:width], ret)
    return ret


def pad_or_truncate(ls, target_len, fill_val):
    return ls[:target_len] + [fill_val] * (target_len - len(ls))


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def gatefun_name(cir, gid):
    gnm = ''
    if cir.is_gate(gid):
        gnm = cir.gatefun(gid).name.lower()
    elif cir.is_inst(gid):
        gnm = str(cir.get_node_attr(gid, circuit.ndattr.CELL_NAME))
    else:
        assert False
    return gnm


gatefuns = dict()
gateind2funs = dict()


def gatefun_index(cir, gid):
    gnm = gatefun_name(cir, gid).lower()

    if '_x' in gnm:
        gnm, ds = gnm.split('_x')

    gnm = str().join([s for s in gnm if s.isalpha()])

    if gnm not in gatefuns:
        num = gatefuns[gnm] = len(gatefuns)
        gateind2funs[num] = gnm
    else:
        num = gatefuns[gnm]

    return num


def gate_encoding0(cir, gid):

    num = gatefun_index(cir, gid)
    enc = [num, cir.numfanins(gid)]
    # print(enc)

    return enc

gate_encoding0.enc_size = 2


def gate_encoding1(cir, gid):

    gnm = gatefun_name(cir, gid).lower()
    num = gatefun_index(cir, gid)

    ds = 0
    if '_x' in gnm:
        ds = int(gnm.split('_x')[1])

    enc = [num, cir.numfanins(gid), ds]

    return enc

gate_encoding1.enc_size = 3

class OlessLutDataset:

    def __init__(self, **kwargs):

        self.dill_file = None
        self.model_file = None
        self.bench_dir = None
        self.model = None
        self.redo = False
        self.no_run = False
        self.decrypt = False
        self.decrypt_selftrain = False
        self.verbose = 0
        self.indiv_softmax = False
        self.categorize_target = False
        self.cutK = 2
        self.cutN = 20
        self.ttsize = 2 ** self.cutK
        self.subG = 10
        self.enc_files = []
        self.batch_size = 10
        self.patience = 5
        self.target_enc_method = 'tt'
        self.neighbor_enc_method = 'ordlist'
        self.gate_encodings_method = 0
        self.ml_method = 'sklearn'
        self.num_epochs = 1000
        self.feature_cenc_count = 0
        self.feature_cenc_size = 0
        self.exec_path = os.path.realpath(__file__)
        self.exec_dir = os.path.dirname(self.exec_path)
        print('exec dir: ', self.exec_dir)
        self.lib_file = os.path.join(self.exec_dir, 'nangate45nm_funs.txt')
        self.cell_lib = liberty.read_liberty_functional(self.lib_file)

        self.__dict__.update(kwargs)

        if self.dill_file is None:
            self.dill_file = './datasets/lutsail/olesslut_k{}_N{}_sG{}_tE{}_nE{}_b{}.txt'.format(self.cutK, self.cutN,
                        self.subG, self.target_enc_method, self.neighbor_enc_method, self.bench_dir.replace('*','_').replace('/','_').replace('.', '_'))
            print('dill file is: ', self.dill_file)
        if self.model_file is None:
            self.model_file = './models/lutsail/olesslut_k{}_N{}_sG{}_cat{}_tE{}_nE{}_b{}/'.format(self.cutK, self.cutN,
                                        self.subG, self.categorize_target, self.target_enc_method, self.neighbor_enc_method, self.bench_dir.replace('*','_').replace('/','_').replace('.', '_'))
            try:
                os.mkdir(self.model_file)
            except FileExistsError:
                pass

        self.X = list()
        self.TT = list()
        self.C = list()
        self.CE = list()
        self.NGs = list()

        if self.neighbor_enc_method == 'ordlist':
            self.cut_neighbor_features_fun = self.cut_neighbor_features0
        else:
            raise Exception('unknown neighborhood encoding method', self.neighbor_enc_method)

        if self.gate_encodings_method == 0:
            self.gate_encoding_fun = gate_encoding0
        elif self.gate_encodings_method == 1:
            self.gate_encoding_fun = gate_encoding1
        else:
            raise Exception('unknown gate encoding method ', self.gate_encodings_method)

        self.neighbor_feature_count = self.gate_encoding_fun.enc_size * self.subG

        if self.decrypt:
            return

        if not self.redo:
            try :
                with open(self.dill_file, 'rb') as fn:
                    self.feature_cenc_count = 0
                    self.feature_cenc_size = 0
                    self.X, self.TT, self.C, self.CE, self.NGs = pickle.load(fn)
                    for ce in self.CE:
                        self.feature_cenc_count = max(self.feature_cenc_count, len(ce))
                        self.feature_cenc_size = max(self.feature_cenc_size, max(ce))
                    self.y_ohe_size = self.feature_cenc_count * self.feature_cenc_size
                return
            except Exception:
                pass

        self.feature_cenc_size = 0
        self.feature_cenc_count = 0

        self.bench_files = glob.glob(self.bench_dir)

        for cir_file in self.bench_files:
            print('processing {} '.format(cir_file, cir_file), end='')
            self.process_file(cir_file)

        self.end_symbol = self.feature_cenc_size + 1
        self.feature_cenc_size += 1

        self.feature_cenc_count += 1
        self.y_ohe_size = self.feature_cenc_count * self.feature_cenc_size

        # fill adj matrics
        if self.target_enc_method == 'adjmat':
            self.pad_adjmat()

        self.X, self.TT, self.C, self.CE, self.NGs = sklearn.utils.shuffle(self.X, self.TT, self.C, self.CE, self.NGs)

        # dill data
        if not self.decrypt:
            with open(self.dill_file, 'wb') as fn:
                pickle.dump((self.X, self.TT, self.C, self.CE, self.NGs), fn)

        print('gate funs: ', gatefuns)

        return

    class GraphDataset(Dataset):
        def __init__(self, graphs, ys, **kwargs):
            self.graphs = graphs
            assert (len(ys) == len(self.graphs))
            for i in range(len(ys)):
                self.graphs[i].y = ys[i]
                #print(self.graphs[i])
            super().__init__(kwargs)

        def read(self):
            return self.graphs

    def pad_adjmat(self):
        # print('filling adjacency matrices\n')
        num_cols = self.feature_cenc_count + 2
        num_rows = self.feature_cenc_count
        self.feature_cenc_count = num_rows * num_cols
        for i, genc in enumerate(self.CE):
            fgenc = [0 for i in range(num_cols * num_rows)]
            for row in range(len(genc)):
                for col in range(len(genc[row])):
                    fgenc[row * num_cols + col] = genc[row][col]
            self.CE[i] = fgenc

    def process_file(self, cir_file):
        
        if cir_file.endswith('.bench'):
            scir = circuit.Circuit(cir_file)
        elif cir_file.endswith('.v'):
            scir = circuit.Circuit()
            scir.read_verilog(cir_file)
            # svcir.write_verilog()
            scir.link_to_library(self.cell_lib)
        else:
            assert False

        self.process_cir(scir)

    def clear_data(self):
        self.X, self.TT, self.CE, self.C = [], [], [], []

    def process_cir(self, scir):
        num_cuts = 0
        for gid in scir.gates_and_instances():
            cuts = ccuts.get_node_cuts(scir, gid, self.cutK, self.cutN)
            # print('all cuts: ', cuts)

            for c, cut in enumerate(cuts):
                # print(cut)
                # ccir = self.build_cut_cir(scir, cut)
                # ccir.write_bench()
                # print(cut[0])
                # print(len(cut[0]))
                if len(cut[0]) == self.cutK:
                    # print('found {}-input cut '.format(len(cut[0])) )
                    cutcir = ccuts.build_cut_cir_gen(scir, cut)
                    # cutcir.write_bench()
                    cut_tt = ccuts.get_truth_table(cutcir)
                    # print('tt', cut_tt)
                    neighfeat = self.cut_neighbor_features_fun(scir, cut)

                    dirmap = dict()
                    neighcir = ccuts.cut_neighbor_circuit(scir, cut, self.subG, dirmap)
                    ngraph = self.neighcir2graph(neighcir, cut_tt, dirmap)
                    # print('\ncut:')
                    # cutcir.write_bench()
                    genc = []
                    if self.target_enc_method == 'tt':
                        genc = [0]
                    elif self.target_enc_method == 'inc':
                        genc = self.encode_circuit_incomplete(cutcir)
                    elif self.target_enc_method == 'adjlist':
                        genc = self.encode_circuit_adjlist(cutcir)
                    elif self.target_enc_method == 'adjmat':
                        genc = self.encode_circuit_adjmat(cutcir)
                    elif self.target_enc_method == 'nauty':
                        genc = self.encode_circuit_nauty(cutcir)
                    # print('genc:', genc)
                    # decir = self.decode_circuit1(genc)
                    # print('\ndecir:')
                    # decir.write_bench()
                    # print(self.feature_cenc_count)
                    self.feature_cenc_count = max(len(genc), int(self.feature_cenc_count))
                    self.feature_cenc_size = np.max(genc) + 1

                    self.X.append(neighfeat)
                    self.TT.append(cut_tt)
                    self.C.append(cutcir)
                    self.CE.append(genc)
                    self.NGs.append(ngraph)

                    num_cuts += 1
        print(' -> found {} cuts of size {}'.format(num_cuts, self.cutK))

    def test0(self):
        cir = circuit.Circuit()
        x0 = cir.add_wire(circuit.ntype.IN)
        x1 = cir.add_wire(circuit.ntype.IN)
        w0 = cir.add_wire(circuit.ntype.INTER)
        y0 = cir.add_wire(circuit.ntype.OUT)
        cir.add_gate_wids(circuit.gfun.AND, [x0, x1], w0)
        cir.add_gate_wids(circuit.gfun.OR, [x0, w0, x1], y0)
        cir.write_bench()
        genc = self.encode_circuit_nauty(cir)
        print(genc)

        cir = circuit.Circuit()
        x1 = cir.add_wire(circuit.ntype.IN)
        x0 = cir.add_wire(circuit.ntype.IN)
        w0 = cir.add_wire(circuit.ntype.INTER)
        y0 = cir.add_wire(circuit.ntype.OUT)
        cir.add_gate_wids(circuit.gfun.AND, [x1, x0], w0)
        cir.add_gate_wids(circuit.gfun.OR, [w0, x1], y0)
        cir.write_bench()
        genc = self.encode_circuit_nauty(cir)
        print(genc)

    def test1(self):
        cir = circuit.Circuit()
        x0 = cir.add_wire(circuit.ntype.IN)
        x1 = cir.add_wire(circuit.ntype.IN)
        w0 = cir.add_wire(circuit.ntype.INTER)
        y0 = cir.add_wire(circuit.ntype.OUT)
        cir.add_gate_wids(circuit.gfun.AND, [x0, x1], w0)
        cir.add_gate_wids(circuit.gfun.OR, [x0, w0, x1], y0)
        cir.write_bench()
        genc = self.encode_circuit_adjlist(cir)
        print(genc)

        dcir = self.decode_circuit_adjlist(genc)
        dcir.write_bench()

        return

    def equalize_xy_lengths(self, L):
        for i in range(len(self.X)):
            self.X[i] = resize_fill_list(self.X[i], L)
            self.C[i] = resize_fill_list(self.C[i], L)
        return

    def encode_circuit_incomplete(self, cutcir):

        rootwid = list(cutcir.outputs())[0]
        rootgid = cutcir.fanin(rootwid)
        assert rootgid is not None
        stack = [rootgid]

        featvec = []

        num_g = 0
        visited = set()
        while len(stack) != 0:
            curg = stack.pop(0)
            visited.add(curg)
            featvec.append(int(cutcir.gatefun(curg)))
            featvec.append(len(cutcir.fanins(curg)))
            for gid in cutcir.gfanings(curg):
                if gid not in visited:
                    stack.append(gid)

        #print(featvec)

        return featvec

    # encode with nauty
    def encode_circuit_nauty(self, cir):

        num_nodes = cir.num_allins() + cir.num_gates
        g = pynauty.Graph(num_nodes, directed=True)

        colors = []
        nid2ind = dict()
        ind = 0
        for wid in cir.allins():
            nid2ind[wid] = ind
            ind += 1
        
        for gid in cir.gates():
            nid2ind[gid] = ind
            nid2ind[cir.fanout(gid)] = ind
            ind += 1
        
        for wid in cir.allins():
            colors.append(set([wid]))

        gfun2colind = dict()
        for fun in circuit.gfun:
            colors.append(set())
            gfun2colind[fun] = len(colors) - 1
            #print(fun)

        for gid in cir.gates():
            gfos = cir.gfanoutgs(gid)
            if len(gfos) != 0:
                dests = [nid2ind[x] for x in gfos]
                g.connect_vertex(nid2ind[gid], dests)
            colors[gfun2colind[cir.gatefun(gid)]].add(nid2ind[gid])

        for wid in cir.allins():
            dests = [nid2ind[x] for x in cir.fanouts(wid)]
            g.connect_vertex(nid2ind[wid], dests)

        g.set_vertex_coloring(colors)

        canlab = pynauty.canon_label(g)
        # print('canlab:', canlab)

        # now traverse using canon labeling
        rootwid = list(cir.outputs())[0]
        rootgid = cir.fanin(rootwid)
        assert rootgid is not None
        stack = [rootgid]

        featvec = []

        visited = set()
        while len(stack) != 0:
            cnid = stack.pop(0)
            visited.add(cnid)
            if cir.is_gate(cnid):
                featvec.append(int(cir.gatefun(cnid)))
                featvec.append(len(cir.fanins(cnid)))
            else:
                featvec.append(canlab[nid2ind[cnid]]+1)
                featvec.append(canlab[nid2ind[cnid]]+1)

            sorted_fanins = sorted(cir.fanins(cnid), key=lambda nid : canlab[nid2ind[nid]])
            for gin in sorted_fanins:
                if gin not in visited:
                    stack.append(gin)

        #cir.write_bench()
        #print('feat:', featvec)

        return featvec

    def encode_circuit_adjlist(self, cutcir):

        rootwid = list(cutcir.outputs())[0]
        rootgid = cutcir.fanin(rootwid)
        assert rootgid is not None

        nid2ind = dict()
        i = 0
        for x in cutcir.allins():
            nid2ind[x] = i
            i += 1

        for gid in cutcir.gates_and_instances():
            nid2ind[gid] = i
            nid2ind[cutcir.fanout(gid)] = i
            i += 1

        featvec = []

        visited = set()
        stack = [rootgid]

        while len(stack) != 0:
            cnid = stack.pop(0)
            visited.add(cnid)

            featvec.append(nid2ind[cnid])
            featvec.append(gatefun_index(cutcir, cnid))
            featvec.append(len(cutcir.fanins(cnid)))

            for win in cutcir.fanins(cnid):
                gin = cutcir.fanin(win)
                if gin is None: # leaf node
                    featvec.append(nid2ind[win])
                elif gin not in visited:
                    featvec.append(nid2ind[win])
                    stack.append(gin)

        # print(featvec)

        return featvec

    def decode_circuit_adjlist(self, featvec):

        cir = circuit.Circuit()
        ind2nid = dict()

        i = 0
        visited = set()
        gates = []
        # parse into list of gate-ast
        while i < len(featvec) - 3:
            gind = featvec[i]
            funind = featvec[i+1]
            numfanins = featvec[i+2]
            #print('gind', gind)
            #print('num fanins,', numfanins)
            gfanins = []
            i += 3
            j = i
            while j < i + numfanins:
                inind = featvec[j]
                gfanins.append(inind)
                j += 1

            i = j
            print('gate ', (gind, funind, numfanins, gfanins))
            gates.append((gind, funind, numfanins, gfanins))

        gate_inds = set()
        in_inds = set()
        for gate in gates:
            gate_inds.add(gate[0])
            ind2nid[gate[0]] = -1
        for gate in gates:
            for gfanin in gate[3]:
                if gfanin not in gate_inds:
                    in_inds.add(gfanin)
                    ind2nid[gfanin] = -1

        indlist = sorted(ind2nid)

        for ind in indlist:
            if ind in in_inds:
                ind2nid[ind] = cir.add_wire(circuit.ntype.IN, 'n{}'.format(ind))
            elif ind == gates[0][0]:
                ind2nid[ind] = cir.add_wire(circuit.ntype.OUT, 'n{}'.format(ind))
            else:
                ind2nid[ind] = cir.add_wire(circuit.ntype.INTER, 'n{}'.format(ind))

        for gate in gates:
            gind, funid, numfanins, gfanins = gate
            fanins = [ind2nid[x] for x in gfanins]
            print(fanins)
            fanout = ind2nid[gind]
            fun = gateind2funs[funid] # circuit.gfun(funid)
            cir.add_gate_wids(fun, fanins, fanout)

        return cir


    def encode_circuit_adjmat(self, cutcir):

        rootwid = list(cutcir.outputs())[0]
        rootgid = cutcir.fanin(rootwid)
        assert rootgid is not None

        nid2ind = dict()
        i = 0
        for x in cutcir.allins():
            nid2ind[x] = i
            i += 1

        for gid in cutcir.gates_and_instances():
            nid2ind[gid] = i
            nid2ind[cutcir.fanout(gid)] = i
            i += 1

        num_nodes = i
        #print(num_nodes)
        num_gate_feat = 2

        featmat = [[0 for i in range(num_nodes + num_gate_feat)] for i in range(num_nodes)]

        visited = set()
        stack = [rootgid]

        while len(stack) != 0:
            cnid = stack.pop(0)
            cind = nid2ind[cnid]
            visited.add(cnid)

            featmat[cind][0] = gatefun_index(cutcir, cnid)
            featmat[cind][1] = len(cutcir.fanins(cnid))

            # start of adj-mat connections

            for win in cutcir.fanins(cnid):
                gin = cutcir.fanin(win)
                featmat[nid2ind[win]][cind + num_gate_feat] = 1
                if gin is not None and gin not in visited:
                    stack.append(gin)

        #print(featmat)

        return featmat


    def neighcir2graph(self, cir, cut_tt, dirmap):

        #cir.write_bench()
        num_nodes = cir.num_gates + cir.num_ins_and_keys() + cir.num_outs() + cir.num_inst
        A = sp.lil_matrix((num_nodes, num_nodes))
        xs = gencs.gate_encoding.enc_size + 1
        X = np.zeros((num_nodes, xs))

        y_graph = cut_tt

        # print('Y is ', Y)
        gind = 0
        gid2index = dict()
        for xid in cir.allins():
            gid2index[xid] = gind
            gind += 1

        for gid in cir.gates_and_instances():
            gid2index[gid] = gind
            gind += 1

        for oid in cir.outputs():
            gid2index[oid] = gind
            X[gind][0] = 0  # -1
            gind += 1

        for xid in cir.allins():
            gi0 = gid2index[xid]
            X[xid][0] = 1
            for gout in cir.fanouts(xid):
                gi1 = gid2index[gout]
                A[gi1, gi0] = 1

        for gid in cir.gates_and_instances():
            gi0 = gid2index[gid]
            X[gi0, 0:xs-1] = gencs.gate_encoding(cir, gid)
            X[gi0, xs-1] = dirmap[gid]
            wout = cir.fanout(gid)
            if cir.is_output(wout):
                gi1 = gid2index[wout]
                A[gi1, gi0] = 1
            else:
                for gout in cir.gfanoutgs(gid):
                    gi1 = gid2index[gout]
                    A[gi1, gi0] = 1

        G = Graph(a=A, x=X, y=y_graph)

        return G

    def cut_neighbor_features0(self, cir, cut):

        cutgids = cut[1]
        feature_vec = []

        num_added_gates = 0
        for glrind, glr in enumerate( ccuts.get_gate_layers(cir, cutgids) ):
            for gid, gdir in glr:
                feature_vec.extend( self.gate_encoding_fun(cir, gid) )
                num_added_gates += 1
                if num_added_gates >= self.subG:
                    break
            if num_added_gates >= self.subG:
                break

        if len(feature_vec) < self.neighbor_feature_count:
            feature_vec.extend([0 for i in range(self.neighbor_feature_count - len(feature_vec))])
        #print(len(feature_vec), self.feature_vec_size)
        #assert(len(feature_vec) == self.feature_vec_size)

        return feature_vec

    def cut_neighbor_features1(self, cir, cut):

        cutgids = cut[1]
        feature_vec = []

        num_added_gates = 0
        for glrind, glr in enumerate( ccuts.get_gate_layers(cir, cutgids) ):
            for gid, gdir in glr:
                wid = cir.fanout(gid)
                try:
                    feature_vec.append(list(cut[0]).index(wid) + 1)
                except ValueError:
                    feature_vec.append(0)
                feature_vec.extend(self.gate_encoding_fun(cir, gid))
                feature_vec.append(glrind)
                feature_vec.append(gdir)
                num_added_gates += 1
                if num_added_gates >= self.subG:
                    break
            if num_added_gates >= self.subG:
                break

        if len(feature_vec) < self.neighbor_feature_count:
            feature_vec.extend([0 for i in range(self.neighbor_feature_count - len(feature_vec))])

        # print(len(feature_vec), self.feature_vec_size)
        assert (len(feature_vec) == self.neighbor_feature_count)
        # print(len(feature_vec))

        return feature_vec

    def corrkey_from_file(self, cir_file):
        with open(cir_file, 'r') as fn:
            for ln in fn:
                if 'corrkey=' in ln:
                    corrkey_str = ln.rstrip().split('=')[-1]
                    # print(corrkey_str)
                    # print([x for x in corrkey_str])
                    corrkey = [int(x) for x in corrkey_str]
                    return corrkey

    def eval_decrypt(self):

        assert self.decrypt
        self.bench_files = glob.glob(self.bench_dir)
        assert len(self.bench_files) == 1

        print('decrypting ', self.bench_files)

        cir_file = self.bench_files[0]
        scir = circuit.Circuit(cir_file)

        corrkey = self.corrkey_from_file(cir_file)
        print('corrkey is ', corrkey)

        if self.decrypt_selftrain:
            mcir = copy.deepcopy(scir)
            for gid in mcir.gates():
                if mcir.gatefun(gid) == circuit.gfun.LUT:
                    mcir.set_gatefun(gid, circuit.gfun.NAND)

            self.clear_data()
            self.process_cir(mcir)

            self.model = None
            self.dolearning()
            assert self.model is not None
        else:
            self.model = tf.keras.models.load_model(self.model_file)

        # use model for decryption
        print('decryption model')
        print(self.model.summary())
        pred_key = []
        for gid in scir.gates():
            if scir.gatefun(gid) == circuit.gfun.LUT:
                cut =  (set(scir.fanins(gid)), {gid}, gid)
                te = self.eval_model(scir, cut)
                pred_key.extend(te[0])

        print('corrkey:', corrkey)
        print('predkey:', pred_key)

        kpa = float(sum([corrkey[i] == pred_key[i] for i in range(len(corrkey))])) / len(pred_key) * 100
        print('kpa : {}'.format(kpa))

        return

    def eval_model(self, scir, cut):
        assert self.model is not None

        if self.ml_method == 'nn':
            feat = self.cut_neighbor_features_fun(scir, cut)
            x = np.zeros((1, len(feat)))
            x[0, 0:len(feat)] = feat[:]
            if self.oheX is not None:
                x = self.oheX.transform(x)
            print('calling model on ', x)
            te = self.model.predict(x)
            print(te)
            te = (te > 0.5) * int(1)
            print(te)
            return te
        elif self.ml_method == 'gcn':
            dirmap = dict()
            neighcir = ccuts.cut_neighbor_circuit(scir, cut, self.subG, dirmap)
            cut_tt = [0 for i in range(self.ttsize)]
            ngraph = self.neighcir2graph(neighcir, cut_tt, dirmap)
            data = self.GraphDataset([ngraph], [cut_tt])
            loader = DisjointLoader(data)
            Y_pred = np.zeros((len(data), self.ttsize))
            Y_test = np.zeros((len(data), self.ttsize))

            inputs_te, y_nodes_te = next(loader)
            print(y_nodes_te.shape)
            pred = self.model(inputs_te, training=False)
            pred = pred.numpy()
            print(pred)
            te = (pred > 0.5) * int(1)
            print(te)

            return te


    def dolearning(self):
        if self.target_enc_method == 'tt':
            self.learn_XtoTT()
        else:
            self.learn_XtoCG()

    def learn_XtoCG(self):
        self.G_pred, self.test_indices = self.train_and_predict_model(self.X, self.CE)

    def learn_XtoTT(self):
        self.TT_pred, self.test_indices = self.train_and_predict_model(self.X, self.TT)

    def train_and_predict_model(self, X, Y):
        if self.ml_method == 'sklearn':
            return self.train_and_predict_model_sklearn(X, Y)
        elif self.ml_method == 'nn':
            return self.train_and_predict_model_nn(X, Y)
        elif self.ml_method == 'nn2':
            return self.train_and_predict_model_nn2(X, Y)
        elif self.ml_method == 'ruledict':
            return self.train_and_predict_model_ruledict(X, Y)
        else:
            raise Exception('unknown machine-learning method {}'.format(self.ml_method))

    def evaluate_prediction(self, Y_test, Y_pred):
        print('Y_pred', Y_pred.shape)
        print('Y_test ', Y_test.shape)
        matches = 0
        amatches = 0
        for i in range(Y_pred.shape[0]):
            if self.verbose >= 1:
                print(Y_test[i], Y_pred[i], np.array_equal(Y_test[i], Y_pred[i]))
            if np.array_equal(Y_test[i], Y_pred[i]):
                matches += 1
            for j in range(Y_pred.shape[1]):
                amatches += int(Y_pred[i, j] == Y_test[i, j])

        print('perfect matches: {} / {} -> {}%'.format(matches, Y_pred.shape[0], float(matches) / Y_pred.shape[0] * 100))
        print('all     matches: {} / {} -> {}%'.format(amatches, Y_pred.shape[0] * Y_pred.shape[1], float(amatches) / (Y_pred.shape[0] * Y_pred.shape[1]) * 100))

    def evaluate_prediction_kpa(self, Y_pred, test_indices):
        print('Y_pred', Y_pred.shape)
        #print('Y_test ', test_indices)
        key_matches = 0
        TT_pred = np.zeros((Y_pred.shape[0], self.ttsize))
        if self.target_enc_method == 'tt':
            TT_pred = Y_pred
        elif self.target_enc_method == 'adjlist':
            for i in range(Y_pred.shape[0]):
                try :
                    print('trying to decode')
                    gcir = self.decode_circuit_adjlist(Y_pred[i])
                    print('gcir', gcir)
                    gcir.write_bench()
                    TT_pred[i] = ccuts.get_truth_table(gcir)
                except Exception as e:
                    print('could not decode', e)
                    pass
        elif self.target_enc_method == 'adjmat':
            print('decoding adjmat')
            exit(1)
            for i in range(Y_pred.shape[0]):
                try :
                    gcir = self.decode_circuit_adjmat(Y_pred[i])
                    TT_pred[i] = ccuts.get_truth_table(gcir)
                except Exception: pass

        for i in range(TT_pred.shape[0]):
            print(TT_pred[i], self.TT[test_indices[i]])
            #print((TT_pred[i] == self.TT[test_indices[i]]).sum())
            key_matches += (TT_pred[i] == self.TT[test_indices[i]]).sum()

        all_ttbits = Y_pred.shape[0] * self.ttsize

        print('KPA: {} / {} -> {}%'.format(key_matches, all_ttbits, float(key_matches) / all_ttbits * 100))

    def train_and_predict_model_gcn(self, NGs, Y):

        if self.categorize_target:
            Y, Ymap = categorize_2d(Y)

        Y = pad_2dlist_to_maxlen(Y)
        Y = np.array(Y)

        oheY = None

        if self.target_enc_method != 'tt' or self.categorize_target:
            Y, oheY = one_hot_encode_mat(Y)
        else:
            self.indiv_softmax = False

        need_argmax = self.categorize_target

        print('individual softmax is ', self.indiv_softmax)

        dataset = self.GraphDataset(NGs, Y)

        num_outs = Y.shape[1]

        split = int(0.9 * Y.shape[0])
        test_indices = range(0, split)
        Y_train, Y_test = Y[:split], Y[split:]
        data_tr, data_te = dataset[:split], dataset[split:]

        loader_tr = DisjointLoader(data_tr, batch_size=self.batch_size, epochs=self.num_epochs)
        loader_te = DisjointLoader(data_te)

        # model = GeneralGNN(output=num_outs, activation='tanh')
        # opt = Adam()
        # model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

        X_in = Input(shape=(dataset.n_node_features,), name='X_in')
        A_in = Input(shape=(None,), sparse=True)
        I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)
        M_1 = GCSConv(32, activation='relu')([X_in, A_in])
        M_2 = GCSConv(32, activation='relu')([M_1, A_in])
        M_3 = GCSConv(32, activation='relu')([M_2, A_in])
        # M, A, I = TopKPool(ratio=0.5)([M, A, I])
        # M = GCSConv(32, activation='relu')([M, A])
        # M, A, I = TopKPool(ratio=0.5)([M, A, I])
        # M = GCSConv(32, activation='relu')([M, A])
        #M = spektral.layers.pooling.GlobalAttentionPool(5)([M_3, I_in])
        M = spektral.layers.pooling.GlobalSumPool()([M_3, I_in])


        if self.categorize_target:
            M = layers.Dense(num_outs, activation='softmax')(M)
            self.model = Model(inputs=[X_in, A_in, I_in], outputs=M)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            #print(M.shape)
            if self.indiv_softmax:
                Ms = []
                Ls = []
                cce = tf.keras.losses.CategoricalCrossentropy()
                def custom_loss(y_true, y_pred):
                    pos = 0
                    losses = []
                    for cat in oheY.categories_:
                        losses.append( cce(y_true[pos:pos+len(cat)], y_pred[pos:pos+len(cat)]) )
                        pos += len(cat)
                    return tf.reduce_sum(losses)

                for cat in oheY.categories_:
                    Ms.append(layers.Dense(len(cat), activation='softmax')(M))
                M = layers.Concatenate()(Ms)

                # M = layers.Dense(Y.shape[1])(M)
                self.model = Model(inputs=[X_in, A_in, I_in], outputs=M)
                # model.compile(loss='mse', optimizer='adam')
                self.model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
            else:
                M = layers.Dense(num_outs, activation='sigmoid')(M)
                self.model = Model(inputs=[X_in, A_in, I_in], outputs=M)
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        try:
            self.model.fit(loader_tr.load(),
                      steps_per_epoch=loader_tr.steps_per_epoch,
                      validation_data=loader_te.load(),
                      validation_steps=loader_te.steps_per_epoch,
                      epochs=self.num_epochs,
                      callbacks=[es])
        except KeyboardInterrupt:
            print('Training interrupted. Saving model\n')

        self.model.save(self.model_file)
        #print(model.summary())

        Y_pred = np.zeros((len(data_te), num_outs))
        Y_test = np.zeros((len(data_te), num_outs))

        print(len(data_te))
        print(loader_te.steps_per_epoch)

        for i, batch_te in enumerate(loader_te):
            inputs_te, y_nodes_te = batch_te
            print(y_nodes_te.shape)
            Y_test[i, :] = y_nodes_te[0][:]
            pred = self.model(inputs_te, training=False)
            print(pred.numpy().shape)
            Y_pred[i, :] = pred.numpy()[:]
            if i == loader_te.steps_per_epoch - 1:
                break

        if oheY is not None:
            Y_pred_o = np.zeros(shape=Y_pred.shape)
            pos = 0
            for cat in oheY.categories_:
                print('at cat ', pos, pos + len(cat))
                Y_pred_amax = np.argmax(Y_pred[:, pos:pos + len(cat)], axis=1)
                for i in range(Y_pred_o.shape[0]):
                    Y_pred_o[i, pos + Y_pred_amax[i]] = 1
                pos += len(cat)

            #print(Y_pred_o)
            Y_pred = Y_pred_o
        else:
            Y_pred = (Y_pred > 0.5) * 1

            # # print(inputs_te)
            # pred = model(inputs_te, training=False)
            # if need_argmax:
            #     Y_pred[i, np.argmax(pred)] = 1
            # else:
            #     Y_pred[i, :] = (pred.numpy()[:] > 0.5) * int(1)
            # print(i, '/', loader_te.steps_per_epoch)
            # if i == loader_te.steps_per_epoch - 1:
            #     break

        self.evaluate_prediction(Y_test, Y_pred)

        # Y_pred = model.predict(loader_te.load(), steps=len(Y_test))
        # callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])

        return [], []

    def train_and_predict_model_ruledict(self, X, Y):
    	
        #X = pad_2dlist_to_maxlen(X)
        #Y = pad_2dlist_to_maxlen(Y)

        #X = np.array(X)
        #Y = np.array(Y)

        #print(X.shape)
        #print(Y.shape)

        split = int(0.8 * len(X))
        X_train = X[:split]
        X_test = X[split:]
        Y_train = Y[:split]
        Y_test = Y[split:]

        Rdict = dict()
        AllYs = dict()
        AllXs = set()
        
        for i in range(len(X_train)):
            xsig = tuple(X_train[i])
            ysig = tuple(Y_train[i])
            AllXs.add(xsig)
            AllYs[ysig] = self.TT[i]
            if xsig not in Rdict:
                Rdict[xsig] = dict()
                Rdict[xsig][ysig] = 1
            else:
                if ysig not in Rdict[xsig]:
                    Rdict[xsig][ysig] = 1
                else:
                    Rdict[xsig][ysig] += 1

        print(len(Rdict))
        
        print('num Xs ', len(AllXs))
        print('num Ys ', len(AllYs))
       
        max_y_count = 0
        all_best_y = None
        YcMap = defaultdict(lambda : 0)
        for i in range(len(X_train)):
            xsig = tuple(X_train[i])
            ysig = tuple(Y_train[i])
            YcMap[ysig] += 1

        for ysig, ycount in YcMap.items():
            if ycount > max_y_count:
                max_y_count = ycount
                all_best_y = AllYs[ysig]
                
        print('best of all Y: ', all_best_y)
        #for y, d in AllYs.items():
        #    print(y, ' -> ', d)

        OrdDict = dict()
        for xsig, ydict in Rdict.items():
            ycounts = [x for k, x in ydict.items()]
            ysigs = [k for k, x in ydict.items()]
            #print(ysigs, ycounts)
            bestY_ind = np.argmax(ycounts)
            #print('best ind', bestY_ind)
            bestY = ysigs[bestY_ind]
            OrdDict[xsig] = list(bestY)

        Y_pred = [] * len(X_test)
        for i, xt in enumerate(X_test):
            try:
                xsig = tuple(xt)
                Y_pred.append(OrdDict[xsig])
            except KeyError:
                Y_pred.append([0])

        match = 0; missing = 0; nmatch = 0
        for i, yt in enumerate(Y_test):
            #print(yt, Y_pred[i])
            if Y_pred[i] != [0] and yt == Y_pred[i]:
                match += 1
            if yt == all_best_y:
                nmatch += 1
        
        print('naive matches: {} / {} -> {}%'.format(nmatch, len(X_test), (float(nmatch))/(len(X_test))*100))
        print('perfect matches: {} / {} -> {}%'.format(match, len(X_test), (float(match))/(len(X_test))*100))

        return [], []

    def train_and_predict_model_nn(self, X, Y):

        X = pad_2dlist_to_maxlen(X)

        if self.categorize_target:
            Y, Ymap = categorize_2d(Y)

        Y = pad_2dlist_to_maxlen(Y)
        # print(Y)

        X = np.array(X)
        Y = np.array(Y)

        print('x', X.shape)
        print('y', Y.shape)

        X, oheX = one_hot_encode_mat(X)
        self.oheX = oheX

        num_cols = Y.shape[1]

        oheY = None

        #if self.target_enc_method != 'tt' or self.categorize_target:
        #_, ycats = one_hot_encode_np_exdim0(Y)
        if self.target_enc_method != 'tt' or self.categorize_target:
            Y, oheY = one_hot_encode_mat(Y)
        else:
            self.indiv_softmax = False


        print(Y.shape)
        num_outs = Y.shape[1]

        #Y = np.array(Y)

        indices = np.arange(len(X))
        split = int(0.8 * len(X))
        X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(X, Y, indices, test_size=0.2, random_state=42)

        #print(X_train.shape)
        #print(Y_train.shape)

        maxX = np.max(X)
        maxX = np.max(X)

        # model = models.Sequential()
        num_ins = X.shape[1]
        num_hidden = int( np.sqrt(num_ins * num_outs) )
        M_in = Input(shape=X.shape[1:], name='X_in')
        M = layers.Dense(num_hidden, activation='relu')(M_in)
        M = layers.Dense(num_hidden, activation='relu')(M)
        M = layers.Dense(num_hidden, activation='relu')(M)
        #M = layers.Conv1D(125, kernel_size=4, activation='relu', input_shape=(X.shape[1],1))(M_in)
        #M = layers.Conv1D(125, kernel_size=4, activation='relu')(M)
        #M = layers.Conv1D(64, kernel_size=4, activation='relu')(M)
        #M = layers.Flatten()(M)
        # M = layers.Conv1D(64, kernel_size=3, activation='relu')(M)
        #
        # #model.add(layers.Embedding(input_dim=self.X_train.shape[1], input_length=self.X_train.shape[1], output_dim=10))
        # model.add(layers.Dense(num_outs * 30, input_shape=X.shape[1:], activation='relu'))
        # #model.add(layers.Flatten())
        # #model.add(layers.Dense(500, input_shape=X_train.shape[1:], activation='relu'))
        # model.add(layers.Dense(num_outs * 30, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(num_outs * 20, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(num_outs * 10, activation='relu'))
        # model.add(layers.Dense(500, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(250, activation='relu'))
        #model.add(layers.Dense(250))

        if self.categorize_target:
            M = layers.Dense(num_outs, activation='softmax')(M)
            self.model = Model(inputs=[M_in], outputs=M)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            #print(oheY.categories_)
            #print(M.shape)
            if self.indiv_softmax:
                Ms = []
                Ls = []
                cce = tf.keras.losses.CategoricalCrossentropy()
                def custom_loss(y_true, y_pred):
                    pos = 0
                    losses = []
                    for cat in oheY.categories_:
                        losses.append( cce(y_true[pos:pos+len(cat)], y_pred[pos:pos+len(cat)]) )
                        pos += len(cat)
                    return tf.reduce_sum(losses)

                for cat in oheY.categories_:
                    Ms.append(layers.Dense(len(cat), activation='softmax')(M))
                M = layers.Concatenate()(Ms)

                # M = layers.Dense(Y.shape[1])(M)
                self.model = Model(inputs=[M_in], outputs=M)
                # model.compile(loss='mse', optimizer='adam')
                self.model.compile(loss=custom_loss, optimizer='adam')
            else:
                M = layers.Dense(num_outs, activation='sigmoid')(M)
                self.model = Model(inputs=[M_in], outputs=M)
                self.model.compile(loss='binary_crossentropy', optimizer='adam')

        print(self.model.summary())

        #print(X_train.shape)
        #print(X_test.shape)
        #print(Y_train.shape)
        #print(Y_test.shape)

        es = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        try:
            self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.num_epochs, batch_size=self.batch_size,
                       callbacks=[es])
        except KeyboardInterrupt as kb:
            print('interrupted. saving model')

        self.model.save(self.model_file)
        #model = tree.DecisionTreeRegressor()

        # indices = np.arange(len(X))
        #sX_train, sX_test, sY_train, sY_test, indices_train, indices_test = train_test_split(sX, sY, indices, test_size=0.2, random_state=42)
        # X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.2, random_state=42)
        #

        #model = MLPRegressor(hidden_layer_sizes=500)
        #model.fit(sX_train, sY_train)

        #print(sX_test.shape)        
        #Y_pred = decode_sequence(sX_test)
        Y_pred = self.model.predict(X_test)

        Y_pred_orig = None
        print('Y_pred', Y_pred.shape)
        print('Y_test ', Y_test.shape)

        if oheY is not None:
            Y_pred_o = np.zeros(shape=Y_pred.shape)
            pos = 0
            for cat in oheY.categories_:
                print('at cat ', pos, pos+len(cat))
                Y_pred_amax = np.argmax(Y_pred[:, pos:pos+len(cat)], axis=1)
                for i in range(Y_pred_o.shape[0]):
                    Y_pred_o[i, pos + Y_pred_amax[i]] = 1
                pos += len(cat)

            print(Y_pred_o)
            Y_pred = Y_pred_o

            Y_pred_orig = oheY.inverse_transform(Y_pred)
        else:
            Y_pred = (Y_pred > 0.5) * 1
            Y_pred_orig = Y_pred


        #Y_pred = (Y_pred > 0.5) * 1
        print('Y_pred', Y_pred.shape)

        self.evaluate_prediction(Y_test, Y_pred)
        #self.evaluate_prediction_kpa(Y_pred_orig, test_indices)
        #print(sklearn.metrics.hamming_loss(Y_test_lbl, Y_pred_lbl))
        #print(sklearn.metrics.accuracy_score(Y_test_lbl, Y_pred_lbl) * 100)

        return Y_pred, test_indices

    def train_and_predict_model_nn2(self, X, Y):

        X = pad_2dlist_to_maxlen(X)
        Y = pad_2dlist_to_maxlen(Y)

        X = np.array(X)
        Y = np.array(Y)

        X, num_xcats = one_hot_encode_np_exdim(X)
        Y, num_ycats = one_hot_encode_np_exdim(Y)

        print(X.shape)
        print(Y.shape)

        indices = np.arange(X.shape[0])
        X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.2,
                                                                                         random_state=42)

        print(X_train.shape)
        print(Y_train.shape)

        num_outs = Y.shape[1] * Y.shape[2]
        model = models.Sequential()
        model.add(layers.Dense(100, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_outs * 10, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(num_outs * 5, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(num_outs * 5, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(num_outs * 5, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(num_outs))
        model.add(layers.Reshape((Y.shape[1], Y.shape[2])))
        print(model.summary())
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)

        es = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        try:
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.num_epochs, batch_size=self.batch_size,
                      callbacks=[es])
        except KeyboardInterrupt as kb:
            print('interrupted. saving model')

        model.save(self.model_file)

        Y_pred = model.predict(X_test)
        print('Y_pred', Y_pred.shape)
        print('Y_test ', Y_test.shape)

        Y_pred = np.argmax(Y_pred, axis=2)
        Y_test = np.argmax(Y_test, axis=2)

        self.evaluate_prediction(Y_test, Y_pred)

        return Y_pred, indices_test

    def train_and_predict_model_sklearn(self, X, Y):

        X = pad_2dlist_to_maxlen(X)

        if self.categorize_target:
            Y, Ymap = categorize_2d(Y)

        Y = pad_2dlist_to_maxlen(Y)

        X = np.array(X)
        Y = np.array(Y)

        self.feature_cenc_size = int(np.max(Y)) + 1
        self.feature_cenc_count = Y.shape[1]

        print('X', X.shape)
        print('Y', Y.shape)

        #X, oheX = one_hot_encode_mat(X)
        Y, oheY = one_hot_encode_mat(Y)
        # X, num_x_cats = one_hot_encode_np_flat(X)
        #Y, num_y_cats = one_hot_encode_np_flat(Y)
        #Y = np.array(Y)

        print('X', X.shape)
        print('Y', Y.shape)
        #print('Y= ', Y)

        indices = np.arange(X.shape[0])
        X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.2,
                                                                                         random_state=42)

        sX_train = X_train
        sX_test = X_test
        sY_train = Y_train
        sY_test = Y_test
        print(Y_test)

        print(X_train.shape)
        print(Y_train.shape)

        #model = tree.DecisionTreeRegressor()
        #model = sklearn.ensemble.RandomForestClassifier()
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=500, max_iter=self.num_epochs, verbose=True)
        # model = MLPRegressor(hidden_layer_sizes=500, max_iter=100)
        model.fit(sX_train, sY_train)

        # print(sX_test.shape)
        # Y_pred = decode_sequence(sX_test)
        Y_pred = model.predict(sX_test)

        Y_pred = np.array(Y_pred)
        #Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], 1))
        print('Y_pred shape', Y_pred.shape)

        #Y_pred = oheY.inverse_transform(Y_pred)
        #Y_test = oheY.inverse_transform(Y_test)

        # Y_pred = one_hot_decode_np_flat(Y_pred, num_y_cats)
        # Y_test = one_hot_decode_np_flat(Y_test, num_y_cats)

        self.evaluate_prediction(Y_test, Y_pred)

        return Y_pred, indices_test


def main(args):

    dataset = OlessLutDataset(**vars(args))

    if args.decrypt:
        dataset.eval_decrypt()
        return

    dataset.dolearning()
    #dataset.learn_backward()
    
    return 

def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser('run_oless_lutsail')


    #args.add_argument('bench_dir', help='circuit file directory or bench-file when evaluating')
    
    args.add_argument('bench_dir', help='bench files glob')

    args.add_argument('-dd', '--data_dir', default='./datasets/sail/', type=str,
                      help='Data directory for saving and loading')

    args.add_argument('-redo', action='store_true')

    args.add_argument('-indsoft', '--indiv_softmax', action='store_true')

    args.add_argument('-decrypt', action='store_true')

    args.add_argument('-decrypt_selftrain', action='store_true')

    args.add_argument('-nep', '--num_epochs', default=1000, type=int,
                      help='number of epochs')

    args.add_argument('-bs', '--batch_size', default=5, type=int,
                      help='batch size')

    args.add_argument('-v', '--verbose', default=0, type=int,
                      help='verbosity level ')

    args.add_argument('-pc', '--patience', default=5, type=int,
                      help='patience value')

    args.add_argument('-mlm', '--ml_method', default='sklearn', type=str,
                      help='machine-learning method')

    args.add_argument('-sG', '--subG', default=5, type=int,
                      help='number of gates in locality')

    args.add_argument('-tE', '--target_enc_method', default='tt', type=str,
                      help='target (missing cut) encoding method : {inc, adjlist, adjmat, nauty}')

    args.add_argument('-catT', '--categorize_target', action='store_true',
                      help='whether to categorize target ')

    args.add_argument('-norun', '--no_run', action='store_true',
                      help='dont run. just report data. ')

    args.add_argument('-K', '--cutK', default=2, type=int,
                      help='K for K-cuts (number of cut inputs)')

    args.add_argument('-N', '--cutN', default=20, type=int,
                      help='N for max number of cuts to explore at each wire')
        
    return args.parse_args()


if __name__=="__main__":
    args = parse_args()
    main(args)
    
    


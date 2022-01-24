
# methods for translating instances to primitives gates in verilog netlists
# for now based on using an instance translation table
# TODO: support liberty libraries
import circuit
from circuit import *
from vparse import *
import re
import tempfile

common_gates_table = """
AND A B Y -> Y = and(A, B)
NAND A B Y -> Y = nand(A, B)
NOR A B Y -> Y = nor(A, B)
OR A B Y -> Y = or(A, B)
BUF A Y -> Y = buf(A)
NOT A Y -> Y = not(A)
XOR A B Y -> Y = xor(A, B)
XNOR A B Y -> Y = xnor(A, B)
DFFSR Q C R S D -> Q = dff(D)
\$_DLATCH_P_ Q E D -> Q = dff(D)
DFF Q C D -> Q = dff(D)
"""

import logging
from lark import Lark, logger

from lark import Transformer
import os
import sys
import time
import re
from circuit import *


def comment_callback(comment):
    #print('found comment ', comment, ' end comment')
    return comment

liberty_parser = Lark(r'''
        
    liberty : node*
    
    ?node : fun_call | node_property | nested_node
    
    fun_call : IDENT "(" qident_list ")" ";"
    node_property : IDENT ":" qident ";"
    nested_node : IDENT "(" IDENT ")" "{" node+ "}"
    
    qident_list : qident ("," qident)*
    qident : QSTRING | IDENT | NUMS
    
    QSTRING : "\"" /[^\"]*/ "\""
    IDENT : /[a-zA-Z\\$_][a-zA-Z0-9\\.$_]*/
    NUMS : /[0-9e.\*\+-]+/
    
    COMMENT.1 : /\/\*(([^\*\/])*(\*(?!\/))*)*\*\//
    
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
    %ignore COMMENT

    ''', start='liberty', parser='lalr', lexer='standard')

def translate_instances(cir, translation_table = common_gates_table):

    instFun2specs = dict()

    for ln in translation_table.split('\n'):
        ln = ln.strip()
        if len(ln.split('->')) == 2:
            # print ln.split('->')
            inst_str, prim_str = ln.split('->')
            inst_specs = inst_str.rstrip().split(' ')
            # print inst_specs
            inst_name = inst_specs[0]
            inst_ports = inst_specs[1:]
            prim_str = prim_str.replace(' ', '')
            prim_list = re.split(r'=|\(|\)|,', prim_str)
            prim_out = prim_list[0]
            prim_fun = prim_list[1]
            prim_fanins = prim_list[2:-1]

            # add to the dict
            instFun2specs[inst_name] = [inst_ports, prim_fun, prim_out, prim_fanins]
            # print [inst_ports, prim_fun, prim_out, prim_fanins]

    to_keep = list()
    for instid in copy.deepcopy(cir.instances()):
        cell_name = cir.gp.nodes[instid]['cell_name']
        port_inds = cir.gp.nodes[instid]['port_inds']
        port_wids = cir.gp.nodes[instid]['port_wids']
        cellobj = cir.cell_mgr.cell_dict[cell_name]

        if cell_name in instFun2specs:
            inst_ports, prim_fun, prim_out, prim_fanins = instFun2specs[cell_name]
            fanin_ids = list()
            fanout_id = -1
            for i in range(len(port_inds)):
                wid = port_wids[i]
                port_name = cellobj.port_names[i]
                assert(port_name in inst_ports)
                if port_name in prim_fanins:
                    fanin_ids.append(wid)
                else:
                    if port_name == prim_out:
                        fanout_id = wid
            # print prim_fun, fanout_id, fanin_ids
            cir.remove_instance(instid)
            cir.add_gate_wids(prim_fun, fanin_ids, fanout_id)

    return


def read_liberty_functional(libfile):

    fn = open(libfile, 'r')

    cell_lib = circuit.CellLibrary()

    lines = list(fn)
    cirfun = None
    cell_name = None
    port_names = []
    port_dirs = []

    for i in range(len(lines)):
        ln = lines[i]
        if 'cell name' in ln or i == len(lines) - 1:
            if cell_name is not None:
                #print('adding cell', cell_name, port_names, port_dirs)
                cell_lib.add_cell(cell_name, list(port_names), list(port_dirs), copy.deepcopy(cirfun))
                port_dirs.clear()
                port_names.clear()
                cirfun = None
                cell_name = None
                #print('\n')
            if i != len(lines) - 1:
                cell_name = ln.split(':')[1].strip()
                #print(cell_name)
        elif 'ports:' in ln:
            ln = ln.replace('ports:{', '')
            ln = ln.replace('}', '')
            ln = ln.split(',')
            for port in ln:
                port = port.strip()
                port_name, port_dir = port.split(':')
                port_name = port_name.strip()
                port_dir = circuit.portdir(int(port_dir))
                #print(port_name, port_dir)
                port_names.append(port_name)
                port_dirs.append(port_dir)
        elif 'cell function ' in ln:
            #print('cir_text: ')
            j = i + 1
            cir_text = []
            while lines[j] != "\n":
                cir_text.append(lines[j])
                j += 1
            #print(cir_text)
            cirfun = circuit.Circuit()
            cirfun.read_bench_from_fn(cir_text)
            #print('read cirfun:')
            #cirfun.write_bench()

    fn.close()

    return cell_lib

if __name__ == '__main__':
    # text = open('./bench/lverilog/rs232.v').read()
    # read_liberty_functional(sys.argv[1])
    text = open(sys.argv[1], 'r').read()
    #print(text)
    read_liberty_functional(sys.argv[1])
    



import sys, copy
import circuit

def get_node_cuts(cir, gid, K, N, mffc=True):
    cuts = []
    S = []
    # cut is tuple (inputs, gates, rootgid)
    gfanins = cir.fanins(gid)

    cuts.append((set(gfanins), set([gid]), gid))
    S.append(len(cuts) - 1)

    while len(S) > 0 and len(cuts) < N:

        ind = S.pop()

        lcut = cuts[ind]

        cutins = lcut[0]
        cutgids = lcut[1]
        rootgid = lcut[2]

        for xid in cutins:
            if mffc:
                not_isolated = False
                if cir.numfanouts(xid) > 1:
                    for fo in cir.fanouts(xid):
                        if fo not in cutgids:
                            not_isolated = True
                            break
                    if not_isolated:
                        continue

            wgid = cir.fanin(xid)
            if wgid is None:
                continue

            newins = copy.deepcopy(cutins)
            wxins = cir.fanins(wgid)
            newins.remove(xid)
            newins = newins.union(wxins)
            if len(newins) <= K:
                newgids = copy.deepcopy(cutgids)
                newgids.add(wgid)
                newcut = (newins, newgids, rootgid)
                cuts.append(newcut)
                S.append(len(cuts) - 1)

        # print(len(cuts))
    return cuts


def build_cut_cir(cir, cut):
    cutcir = circuit.Circuit()
    wout = cir.fanout(cut[2])
    wmap = dict()
    for xin in cut[0]:
        wmap[xin] = cutcir.add_wire(circuit.ntype.IN, cir.name(xin))

    for gid in cut[1]:
        wid = cir.fanout(gid)
        wmap[wid] = cutcir.add_wire(circuit.ntype.OUT if wid == wout else circuit.ntype.INTER)

    for gid in cut[1]:
        nfanins = [wmap[gin] for gin in cir.fanins(gid)]
        nfanout = wmap[cir.fanout(gid)]
        cutcir.add_gate_wids(cir.gatefun(gid), nfanins, nfanout)

    return cutcir
        
def build_cut_cir_gen(cir, cut):
    cutcir = circuit.Circuit()
    cutcir.cell_mgr = cir.cell_mgr
    wout = cir.fanout(cut[2])
    wmap = dict()
    for xin in cut[0]:
        wmap[xin] = cutcir.add_wire(circuit.ntype.IN, cir.name(xin))

    for gid in cut[1]:
        #if cir.is_inst(gid):
        #print(cir.instance_str_bench(gid))
        for wid in cir.fanouts(gid):
            wmap[wid] = cutcir.add_wire(circuit.ntype.OUT if wid == wout else circuit.ntype.INTER)

    for gid in cut[1]:
        if cir.is_gate(gid):
            nfanins = [wmap[gin] for gin in cir.fanins(gid)]
            nfanout = wmap[cir.fanout(gid)]
            cutcir.add_gate_wids(cir.gatefun(gid), nfanins, nfanout)
        elif cir.is_inst(gid):
            port_wids = cir.get_node_attr(gid, circuit.ndattr.PORT_WIDS)
            port_inds = cir.get_node_attr(gid, circuit.ndattr.PORT_INDS)
            cell_name = cir.get_node_attr(gid, circuit.ndattr.CELL_NAME)
            cut_pwids = []; cut_pinds = port_inds
            for i in range(len(port_wids)):
                cut_pwids.append(wmap[port_wids[i]])
            cutcir.add_instance_wids(cir.name(gid), cell_name, cut_pwids, cut_pinds)

    return cutcir


def get_truth_table(cir):
    out_tt = []

    assert cir.num_allins() < 12
    assert cir.num_outs() == 1

    oid = list(cir.outputs())[0]

    tt = 0
    max_tt = 2 ** cir.num_allins()
    for tt in range(max_tt):
        smap = dict()
        for i, xid in enumerate(cir.allins()):
            smap[xid] = ((tt >> i) & 1 == 0)
        cir.simulate(smap)
        out_tt.append(int(smap[oid]))

    return out_tt
        
def get_gate_layers(cir, gids):

    gate_visited = set(gids)
    gate_layers = [[(x, 0) for x in gids]]
    

    while True:
            
        next_gate_layer = []
        
        for d in [0, 1]:            
            for gid, gdir in gate_layers[-1]:  
                for gid1 in cir.faninfanins(gid) if d else cir.fanoutfanouts(gid):
                    if gid1 not in gate_visited:
                        next_gate_layer.append((gid1, d))
                        gate_visited.add(gid1)
        
        if len(next_gate_layer) != 0:
            gate_layers.append(next_gate_layer)
            yield gate_layers[-1]
        else:
            break


def cut_neighbor_circuit(cir, cut, num_subg_gates, neighdir=None):

    cutgids = cut[1]
    feature_vec = []
    neighgates = set()
    neighcir = circuit.Circuit()
    neighcir.cell_mgr = cir.cell_mgr

    if neighdir is None:
        neighdir = dict()

    cirdirmap = dict()

    num_added_gates = 0
    for glr in get_gate_layers(cir, cutgids):
        for gid, gdir in glr:
            neighgates.add(gid)
            cirdirmap[gid] = gdir
            num_added_gates += 1
            if num_added_gates >= num_subg_gates:
                break
        if num_added_gates >= num_subg_gates:
            break

    wmap = dict()

    for cutin in cut[0]:
        wmap[cutin] = neighcir.add_wire(circuit.ntype.INTER, cir.name(cutin))

    cutout = cir.fanout(cut[2])
    wmap[cutout] = neighcir.add_wire(circuit.ntype.INTER, cir.name(cutout))


    ngid = neighcir.add_gate_wids(circuit.gfun.SUM, [wmap[wid] for wid in cut[0]], wmap[cutout])
    neighdir[ngid] = 0

    for gid in neighgates:
        for wid in cir.neighbors(gid):
            if wid not in wmap:
                wmap[wid] = neighcir.add_wire(circuit.ntype.INTER, cir.name(wid))
        if cir.is_gate(gid):
            nfanins = [wmap[gin] for gin in cir.fanins(gid)]
            nfanout = wmap[cir.fanout(gid)]
            ngid = neighcir.add_gate_wids(cir.gatefun(gid), nfanins, nfanout)
            neighdir[ngid] = cirdirmap[gid]
        elif cir.is_inst(gid):
            port_wids = cir.get_node_attr(gid, circuit.ndattr.PORT_WIDS)
            port_inds = cir.get_node_attr(gid, circuit.ndattr.PORT_INDS)
            cell_name = cir.get_node_attr(gid, circuit.ndattr.CELL_NAME)
            port_names = cir.cell_mgr.cell_dict[cell_name].port_names
            cut_port_ids = [wmap[x] for x in port_wids]
            cut_port_name_pairs = [(port_names[i], neighcir.name(cut_port_ids[i])) for i in range(len(port_names))]

            ngid = neighcir.add_instance(cir.name(gid), cell_name, cut_port_name_pairs)
            neighdir[ngid] = cirdirmap[gid]


    #print(len(feature_vec), self.feature_vec_size)
    #assert(len(feature_vec) == self.feature_vec_size)

    # print('final neighcir')
    # neighcir.write_verilog()

    return neighcir


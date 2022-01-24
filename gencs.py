
import sys
import circuit


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

def gate_encoding(cir, gid):

    enc = []
    gnm = gatefun_name(cir, gid).lower()
    ds = 0

    if '_x' in gnm:
        gnm, ds = gnm.split('_x')
        ds = int(ds)

    if gnm not in gatefuns:
        num = gatefuns[gnm] = len(gatefuns)
    else:
        num = gatefuns[gnm]

    enc.append(num)

    # encode num inputs
    enc.append(cir.numfanins(gid))

    enc.append(int(ds))
    #print(enc)

    return enc


gate_encoding.enc_size = 3

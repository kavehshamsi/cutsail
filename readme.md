
This is a repository for CUTSAIL, which is an oracle-less attack against look-up-table locking. CUTSAIL is effectively a machine-learning based k-cut prediction algorithm. Given the vicinity of a missing k-cut it tries to predict its functionality. The paper appears in GLSVLSI22: 

[Kaveh Shamsi, and Guangwei Zhao, "An Oracle-Less Machine-Learning Attack against Lookup-Table-based Logic Locking". ACM Great Lakes Symposium on VLSI (GLSVLSI) (2022).](https://personal.utdallas.edu/~kaveh.shamsi/publications/GLSVLSI22_CUTSAIL.pdf)

The following command will train a 2-cut prediction model on the combinational ISCAS benchmarks (starting with 'c') in the bench/iscas_abc_simple/ directory:

python3.9 ./oless_lut_master.py './bench/iscas_abc_simple/c*'


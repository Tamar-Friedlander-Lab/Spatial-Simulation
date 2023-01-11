# Spatial-Simulation
$I_j^i(x):=$ interactions of female $i$ and male $j$.\
Computed as
$I_j^i(\bar{x})=N_i(\bar{x})\cdot \frac{(f*N_j)(\bar{x})}{\sum_{k\neq i}(f*N_k)(\bar{x})}$ \
We can compute the next generation as\
$\hat{N_i}(\bar{x}) = C(\frac{1}{2}(\sum_jI_j^i(\bar{x}) + \sum_iI_j^i(\bar{x})))=C(\frac{1}{2}(I^i(\bar{x})+I_i(\bar{x})))$
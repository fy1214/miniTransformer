这个测试目录的建立的目的是为了测试miniTransformer的并行化策略是否正确，即TP-SP策略


test_linear_tp.py
执行方式：torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2 --no-sp

假设tp=1的case下：
forward：
x=[M, K], w=[N, K]，forward output o=x @ w=[M, N]

backward:
dy=[M, N], dx=dy @ w.T=[M, K], dw=dy.T @ x.T=[N,K]

第一类：只开TP=n的情况，且是columnwise的情况:
1、不开启的sequence_parallel

test_column_parallel_forward: sequence_parallel=False
x_i = [M, K], w_i=[N//n, K], o_i=[M, N//n], 且 o = [o_0, o_1, ..., o_n-1]
所以在实现的时候，我们要保证inp_tp是input_full的全部，且linear_tp的weight是linear_baseline的1/n部分
我们需要验证的点就在于，最后的o 是否等于 [o_0, o_1, ..., o_n-1]

因为紧跟着column_parallel linear的是 row parallel linear，所以在linear中并没有主动对算出来的o_i做gather，因此需要我们手动测试o = [o_0, o_1, ..., o_n-1]


test_column_parallel_backward：sequence_parallel=False，此时
x_i = [M, K], w_i=[N//n, K]
dy_i = [M, N // n], dx_i = dy_i @ w_i = [M, K], dw_i=dy_i.T @ x_i.T=[N//n, K], 且dx=dx_0+dx_1+...+dx_n-1, dw=[dw_0, dw_1,...,dw_n-1]

我们需要验证的点就在于，最后的dw=[dw_0, dw_1,...,dw_n-1]，且dx=dx_0+dx_1+...+dx_n-1

对于dx，linear会在计算完dx_i后通过allReduce同步成dx，因此我们只需要校验每个rank上的x_i.grad是否等于x.grad即可
对于dw，linear不需要做dw的同步，因此也需要我门手动测试w.grad=[w_0.grad,w_1.grad,w_2.grad,...,w_n-1.grad]

2、开启了的sequence_parallel

test_column_parallel_forward：sequence_parallel=True
与没开启sequence_parallel相比，linear在forward的时候多做了一步计算，即在计算gemm前会先把input gather起来，其余步骤和sequence_parallel=False相同

因此在测试sp的时候我们只需要考虑输入切分成x_i=[M//n, K], 然后和sequence_parallel=False一样比较结果即可

test_column_parallel_backward：sequence_parallel=True

由于sp的存在，我们在sequence_parallel=False时计算出来的dx需要拆开成x_0,x_1,...,x_n-1份scatter到每个tp-rank上。因此在linear中在计算出dx后，需要额外做一次
dx的reduce-scatter操作，此时出来的dx_i校验，对于每个rank需要校验dx[i,:]=dx_i

sp的存在，不影响dw的计算，因此校验方式与sequence_parallel=False一样。


第二类：只开TP=n的情况，且是rowwise的情况

1、不开启的sequence_parallel
test_row_parallel_forward: sequence_parallel=False

在row parallel的情况下，此时w_i=[N, K//n]，对应的x也必须做纵切：x_i=[M, K//n]
所以每个rank下计算的o_i=[N, M]，且o=o_0+o_1+...+o_n-1
linear中在计算完o后，做一次reduce，因此我们只需要检验每个rank计算得到的o_i=o 即可。

test_row_parallel_backward：sequence_parallel=False

row parallel的情况下，此时dy_i = [M, N], dx_i = dy_i @ w_i = [M, K//n], dw_i=dy_i.T @ x_i=[N, K//n]

sp未开启的情况下，由于forward中做了reduce，此时拿到的dy_i应该等于dy，我们只需要校验计算出来的dx_i=dx[M, i], dw_i=[N, K//n]


2、开启了的sequence_parallel
test_row_parallel_forward: sequence_parallel=True

sp不影响x的横切，因此还是x_i=[M, K//n]，w_i=[N, K//n]，每个rank下计算的o_i=[N, M]，且o=o_0+o_1+...+o_n-1
由于计算完o后，我们需要将输出o拆分成sp，因此在linear中会对o做reduce scatter，切分到每一个rank(dim=0)上。此时o=[o_0,o_1,...,o_n-1]
所以sp下我们应该验证linear输出o_i = o[i, :]

test_row_parallel_backward：sequence_parallel=True

开启sp的情况下，我们得到的每个dy_i=[M//n, N]，因此在linear中我们会对dy做allgather获得完整的[M, N]，剩下的计算与sequence_parallel=False一致。


test_te_linear_nvfp4_tp.py
与上面测试逻辑一致，只是换成了测试Transformer上linear的tp-sp正确性。
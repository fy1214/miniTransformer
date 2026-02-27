import pytest
import torch
from quartet2.linear import Quartet_II_linear
from miniTransformer.module.linear import Linear

torch._dynamo.config.suppress_errors = True

def build_linear(
    in_features=None,
    out_features=None
):
    in_features = in_features
    out_features = out_features
    sequence_parallel = False
    fuse_wgrad_accumulation = False
    tp_group = None
    tp_size = 1
    get_rng_state_tracker = None
    rng_tracker_name = None
    init_method = None
    bias = False
    return_bias = False
    params_dtype = torch.bfloat16
    parallel_mode = 'column'
    parameters_split = None
    device = "cuda"
    ub_overlap_rs = False
    ub_overlap_ag = False
    ub_name = None
    # JQ: hybrid additional args
    layer_number = 0

    linear = Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=device,
        params_dtype=torch.bfloat16,
    )
    # mock main_grad
    linear.weight.main_grad = torch.zeros_like(linear.weight.data, dtype=torch.float32)
    return linear

def ref_in_fp32_mini(H, x, y, head, w1):
    w1_ref = build_linear(H, H)
    with torch.no_grad():
        w1_ref.weight[...] = w1
    hid = w1_ref(x, is_first_microbatch=True, fp8_output=False, inp_meta=None)
    loss = (hid @ head - y).pow(2).mean()
    print(f'mini loss:{loss}')
    loss.backward()
    return hid, w1_ref.weight.grad.float().detach(), x.grad.float().detach()



def ref_in_fp32(H, x, y, head, w1):
    w1_ref = Quartet_II_linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        w1_ref.weight[...] = w1
    hid = w1_ref(x)
    loss = (hid @ head - y).pow(2).mean()
    loss.backward()
    return hid, w1_ref.weight.grad.float().detach(), x.grad.float().detach()


def eval(res, ref):
    with torch.no_grad():
        res = res.float()
        ref = ref.float()
        quad_err = (res - ref).pow(2).mean() / ref.pow(2).mean()
        eff_bitwidth = (-torch.log2(quad_err) / 2)
        cosine = (res.flatten() @ ref.flatten()) / (ref.flatten() @ ref.flatten())
        return quad_err.item(), eff_bitwidth.item(), cosine.item()


def print_error(name, quad_err, eff_bitwidth, cosine):
    print(f"{name}: {quad_err:.4f} mrqe; {eff_bitwidth:.2f} bits; {cosine:.3f} cosine")


def evaluate_accuracy(B, T, H, steps):
    x = torch.randn((B, T, H), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = torch.randn((B, T, 1), device='cuda', dtype=torch.bfloat16)

    W1 = Quartet_II_linear(H, H, device='cuda', dtype=torch.bfloat16)

    with torch.no_grad():
        W1.weight /= (H**0.5 * W1.weight.std())

    head = torch.randn(H, 1, device='cuda', dtype=torch.bfloat16)

    hid_mini, w1_ref_grad_mini, x_ref_grad_mini = ref_in_fp32_mini(H, x, y, head, W1.weight)
    #hid_mini, w1_ref_grad_mini, x_ref_grad_mini = ref_in_fp32(H, x, y, head, W1.weight)

    hid = x @ W1.weight.T
    loss = (hid @ head - y).pow(2).mean()
    print(f'loss:{loss}')
    W1.weight.grad = None
    x.grad = None
    X1 = torch.zeros_like(x, dtype=torch.float32)
    R1 = torch.zeros_like(W1.weight, dtype=torch.float32)
    for _ in range(steps):
        loss.backward()
        R1 += W1.weight.grad.float()
        X1 += x.grad.float()
        W1.weight.grad = None
        X1.grad = None

    with torch.no_grad():
        R1 /= steps

        hid_err = eval(hid, hid_mini)
        #hid_err = eval(hid, hid_ref)
        print_error("hid", *hid_err)

        #w1_err = eval(R1, w1_ref_grad)
        w1_err = eval(R1, w1_ref_grad_mini)
        print_error("W1", *w1_err)

        x1_err = eval(X1, x_ref_grad_mini)
        print_error("X1", *x1_err)

    return w1_err


torch.manual_seed(42)

@pytest.mark.parametrize(
    "steps, expected",
    [(1, (1.9, 2.5, 3.8, 0.06))])
     #(4, (2.9, 3.4, 4.0, 0.02)),
     #(16, (3.9, 4.5, 6.4, 0.01)),
     #(64, (4.9, 5.5, 7.5, 0.01)),
     #(256, (5.9, 6.5, 8.1, 0.005)),
     #(1024, (6.9, 7.5, 8.9, 0.002)),
     #(4096, (7.6, 8.2, 8.9, 0.001))])
def test_backward_accuracy(steps, expected):
    B = 8
    T = 16
    H = 256
    print(f"{steps=}:")
    w1e = evaluate_accuracy(B, T, H, steps)

    # mrqe
    assert 0.01 < w1e[0] < 0.05

    # bits
    assert  w1e[1] >= 2.3

    # cosine
    assert 0.95 < w1e[2] < 1.05


def test_compile_fwd():
    W = build_linear(128, 256)
    def fwd(x): return W(x)
    torch.compile(fwd, fullgraph=True)(torch.randn(1, 128, 128, device='cuda', dtype=torch.bfloat16))


def test_result_fwd():
    W = build_linear(128, 256)
    x = torch.randn((1, 512, 128), device="cuda", dtype=torch.bfloat16)
    y = W(x)
    y_ref = x @ W.weight.T
    se, bits, cos = eval(y, y_ref)
    assert cos > 0.99
    assert bits > 2.8

if __name__ == '__main__':
    test_backward_accuracy(1,[1.9, 2.5, 3.8, 0.06])
# Custom profile override
import os
import pathlib
import re
import subprocess
import tempfile
import git
from addict import Dict
import importlib.util
from thop import profile as th_profile
import contextlib

MAESTRO_ENV_KEY = "USE_MAESTRO"


def set_maestro(on_off: bool):
    os.environ[MAESTRO_ENV_KEY] = "1" if on_off is True else "0"


def is_maestro():
    return os.environ.get(MAESTRO_ENV_KEY, "0") == "1"


@contextlib.contextmanager
def with_maestro(on_off: bool):
    prev = is_maestro()
    set_maestro(on_off)
    yield None
    set_maestro(prev)


def model_summary(model, inputs, batch_size=-1, skip_classes=tuple()):
    summary = {}

    def _hook(m, ip, op):
        cl = str(m.__class__).split(".")[-1].split("'")[0]
        keys = [x.__module__ for x in m.__class__.__mro__]
        keys = list(filter(lambda k: k.endswith("conv") or k.endswith("linear"), keys))
        if len(keys) == 0:
            # only process conv and linear modules
            # print(f"skipping {cl}, {keys}")
            return
        keys = "Conv" if keys[0].endswith("conv") else "Linear"
        idx = len(summary)
        sk = f"{keys}-{cl}-{idx+1}"
        # print(f"processing {cl}")
        params = list(m.weight.size())
        if m.bias is not None:
            params += list(m.bias.size())
        n_p = 1
        for p in params:
            n_p *= p
        summary[sk] = {
            "type": "CONV",
            "input_shape": [batch_size] + list(ip[0].size())[1:],
            "output_shape": [batch_size] + list(op.size())[1:],
            "nb_params": n_p,
            "trainable": m.weight.requires_grad,
        }
        if len(m.weight.size()) == 4:
            groups = m.groups
            _, C, Y, X = ip[0].size()
            _, K, Yo, Xo = op.size()
            _, _, R, S = m.weight.size()
            summary[sk]["stride"] = m.stride
            if groups == C:
                summary[sk]["type"] = "DSCONV"
                K = 1
        else:
            K, C = m.weight.size()
            X, Xo, Y, Yo, R, S = 1, 1, 1, 1, 1, 1
            summary[sk]["stride"] = None
        summary[sk]["dimension_ic"] = (batch_size, K, C, R, S, Y, X)
        summary[sk]["dimension_oc"] = (batch_size, K, C, R, S, Yo, Xo)

    def _apply(m):
        """
        Recursively apply hook to aplicable modules.
        Returns list of all hooks in dfs order.
        """
        if isinstance(m, skip_classes):
            # skip for this module
            return []
        rt = [_apply(c) for c in m.children()]
        rt = [item for sublist in rt for item in sublist]
        return [m.register_forward_hook(_hook)]

    # print(f"skip_classes: {skip_classes}")
    hks = _apply(model)
    model(*inputs)
    for h in hks:
        h.remove()
    return summary


# import maestro code
def get_maestro(clone_dir):
    # return custom object
    ret = Dict()
    # store paths
    ret.dir = clone_dir
    ret.path = pathlib.Path(clone_dir.name)
    print(f"Cloning maestro at: {ret.path}")
    git.Git(ret.path).clone("git://github.com/maestro-project/maestro.git")
    # import stuff
    print("Importing maestro code...")
    spec = importlib.util.spec_from_file_location(
        "summary",
        ret.path
        / "maestro"
        / "tools"
        / "frontend"
        / "helpers"
        / "torch_maestro_summary.py",
    )
    ret.summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ret.summary)
    ret.summary = ret.summary.summary
    return ret


maestro = get_maestro(tempfile.TemporaryDirectory())


def make_model_file(layer, inputs, filename, custom_ops=tuple()):
    mae_summary = model_summary(
        layer, inputs, batch_size=inputs[0].shape[0], skip_classes=custom_ops
    )
    # print(f"mae_summary: {mae_summary}")
    with open(filename, "w") as fo:
        fo.write("Network {} {{\n".format(layer.__module__))
        for key, val in mae_summary.items():
            pc = re.compile("^Conv")
            pl = re.compile("^Linear")
            match_pc = pc.match(key)
            match_pl = pl.match(key)
            if match_pc or match_pl:
                fo.write("Layer {} {{\n".format(key))
                type = val["type"]
                fo.write("Type: {}\n".format(type))
                if not match_pl:
                    fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                fo.write(
                    "Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                        *val["dimension_ic"][1:]
                    )
                )
                fo.write("}\n")
        fo.write("}")


def parse_result(res):
    """
    Returns total energy/throughput.
    """
    ret = 0.0
    for line in filter(lambda s: s != "", res.split("\n")):
        # parse key,value
        ln = [x.strip() for x in line.split(":")]
        if len(ln) > 1:
            k, v = ln
            if k == "Performance per MAC energy":
                ret += 1 / float(v.split(" ")[0])
    return ret


def profile(layer, inputs=None, custom_ops={}):
    if os.environ.get("USE_MAESTRO", "0") == "1":
        data_folder = maestro.path / "maestro" / "data"
        model_file = data_folder / "model" / "op.m"
        custom_ops = {
            k: v for k, v in custom_ops.items() if v.__name__ == "count_custom"
        }
        make_model_file(layer, inputs, model_file, custom_ops=tuple(custom_ops.keys()))
        # make mapping file
        cd_path = maestro.path / "maestro" / "tools" / "frontend"
        mapping_file = data_folder / "mapping" / "op_map.m"
        os.system(
            f"cd {cd_path} && python modelfile_to_mapping.py --model_file {model_file.name} --dataflow kcp_ws --outfile {mapping_file.name} 1> /dev/null"
        )
        # run maestro
        result = subprocess.check_output(
            f"cd {maestro.path.resolve()} && maestro --HW_file='{data_folder.resolve()}/hw/accelerator_1.m' --Mapping_file='{mapping_file.resolve()}' --print_res=true --print_res_csv_file=true --print_log_file=false",
            shell=True,
        ).decode("utf-8")
        # parse result
        return parse_result(result), {}
    else:
        return th_profile(layer, inputs=inputs, custom_ops=custom_ops)

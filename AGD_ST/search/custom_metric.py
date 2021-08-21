# Custom profile override
import os

if os.environ["USE_MAESTRO"] == "1":
    import logging
    import pathlib
    import re
    import subprocess
    import sys
    import tempfile
    import git
    from addict import Dict

    # import maestro code
    def get_maestro(clone_dir):
        # return custom object
        ret = Dict()
        # store paths
        ret.dir = clone_dir
        ret.path = pathlib.Path(clone_dir.name)
        logging.info(f"Cloning maestro at: {ret.path}")
        git.Git(ret.path).clone("git://github.com/maestro-project/maestro.git")
        sys.path.append(ret.path)
        # import stuff
        logging.info("Importing maestro code...")
        from maestro.tools.frontend.helpers.torch_maestro_summary import summary

        ret.summary = summary
        return ret

    maestro = get_maestro(tempfile.TemporaryDirectory())

    def make_model_file(layer, inputs, filename):
        mae_summary = maestro.summary(layer, inputs)
        with open(filename, "w") as fo:
            fo.write("Network {} {{\n".format(model.__module__))
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
        for line in filter(lambda s: s is not "", res.split("\n")):
            # parse key,value
            ln = [x.strip() for x in line.split(":")]
            if len(ln) > 1:
                k, v = ln
                if k == "Performance per MAC energy":
                    ret += 1 / float(v.split(" ")[0])
        return ret

    def profile(layer, inputs=None, custom_ops={}):
        data_folder = maestro.path / "maestro" / "data"
        model_file = data_folder / "model" / "op.m"
        make_model_file(layer, inputs, model_file)
        # make mapping file
        cd_path = maestro.path / "maestro" / "tools" / "frontend"
        mapping_file = data_folder / "mapping" / "op_map.m"
        os.system(
            f"cd {cd_path} && python modelfile_to_mapping.py --model_file {model_file.name} --dataflow kcp_ws --outfile {mapping_file.name}"
        )
        # run maestro
        result = subprocess.check_output(
            f"cd {maestro.path.resolve()} && maestro --HW_file='{data_folder.resolve()}/hw/accelerator_1.m' --Mapping_file='{mapping_file.resolve()}' --print_res=true --print_res_csv_file=true --print_log_file=false",
            shell=True,
        ).decode("utf-8")
        # parse result
        return parse_result(result), {}


else:
    from thop import profile

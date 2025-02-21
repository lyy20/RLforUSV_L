import importlib.util
# import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location('', pathname)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    # return imp.load_source('', pathname)
    # #原imp代码，但imp被py3.4后废除改为importlib替代
    return my_module

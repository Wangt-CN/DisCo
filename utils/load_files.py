import json
import os
import os.path as op
import errno
import yaml
from collections import OrderedDict
try:
    from azfuse import File
except ImportError:
    print("azfuse is not installed")


def load_labelmap_file(labelmap_file):
    label_dict = None

    if labelmap_file.endswith('json'):
        label_dict = json.load(open(labelmap_file, 'r'))['label_to_idx']
        label_dict = {key: val-1 for key, val in label_dict.items()}
        return label_dict

    if labelmap_file is not None and op.isfile(labelmap_file):
        label_dict = OrderedDict()
        with open(labelmap_file, 'r') as fp:
            for line in fp:
                label = line.strip().split('\t')[0]
                if label in label_dict:
                    raise ValueError("Duplicate label " + label + " in labelmap.")
                else:
                    label_dict[label] = len(label_dict)
    return label_dict


def config_dataset_file(data_dir, dataset_file):
    if dataset_file:
        if op.isfile(dataset_file):
            dataset_file = dataset_file
        elif op.isfile(op.join(data_dir, dataset_file)):
            dataset_file = op.join(data_dir, dataset_file)
        else:
            raise ValueError("cannot find file: {}".format(dataset_file))
    return dataset_file


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def load_box_linelist_file(linelist_file):
    if linelist_file is not None:
        img_line_list = []
        box_line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                idx = [int(_) for _ in i.strip().split('\t')]
                img_line_list.append(idx[0])
                box_line_list.append(idx[1])
        return [img_line_list, box_line_list]


def load_from_yaml_file(yaml_file):
    try:
        from azfuse import File
        with File.open(yaml_file, 'r') as fp:
            return yaml.load(fp.read(), Loader=yaml.CLoader)
    except ImportError:
        print("azfuse is not installed")
        with open(yaml_file, 'r') as fp:
            return yaml.load(fp, Loader=yaml.CLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )

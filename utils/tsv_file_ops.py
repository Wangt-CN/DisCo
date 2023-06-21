import os
import os.path as op
import json
import numpy as np
import base64
import cv2
import math
from tqdm import tqdm

from .misc import load_from_yaml_file, write_to_yaml_file
from .tsv_file import TSVFile
from .misc import ensure_directory, exclusive_open_to_read


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except ValueError:
        return None


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with exclusive_open_to_read(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    if not is_py2:
        sep = sep.encode()
    with open(tsv_file_name_tmp, 'wb') as fp, open(
            tsv_lineidx_file_tmp, 'w') as fpidx, open(
                tsv_8b_file_tmp, 'wb') as fp8b:
        assert values is not None
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(map(
                        lambda v: v.encode('utf-8')
                        if isinstance(v, unicode)
                        else str(v), value)) + '\n'
            else:
                value = map(
                    lambda v: v
                    if type(v) == bytes
                    else str(v).encode(),
                    value)
                v = sep.join(value) + b'\n'
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            # although we can use sys.byteorder to retrieve the system-default
            # byte order, let's use little always to make it consistent and
            # simple
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    os.rename(tsv_8b_file_tmp, tsv_8b_file)
    os.rename(tsv_file_name_tmp, tsv_file_name)
    assert os.path.exists(tsv_file_name)


def generate_lineidx_file(tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    tsv_8b_file = tsv_lineidx_file + '.8b'
    idx = 0
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    tsv_8b_file_tmp = tsv_8b_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    if not is_py2:
        sep = sep.encode()
    values = tsv_reader(tsv_file_name)
    with open(
            tsv_lineidx_file_tmp, 'w') as fpidx, open(
                tsv_8b_file_tmp, 'wb') as fp8b:
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(map(
                        lambda v: v.encode('utf-8')
                        if isinstance(v, unicode)
                        else str(v), value)) + '\n'
            else:
                value = map(
                    lambda v: v
                    if type(v) == bytes
                    else str(v).encode(),
                    value)
                v = sep.join(value) + b'\n'
            fpidx.write(str(idx) + '\n')
            # although we can use sys.byteorder to retrieve the system-default
            # byte order, let's use little always to make it consistent and
            # simple
            fp8b.write(idx.to_bytes(8, 'little'))
            idx = idx + len(v)
    # the following might crash if there are two processes which are writing at
    # the same time. One process finishes the renaming first and the second one
    # will crash. In this case, we know there must be some errors when you run
    # the code, and it should be a bug to fix rather than to use try-catch to
    # protect it here.
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    os.rename(tsv_8b_file_tmp, tsv_8b_file)
    assert os.path.exists(tsv_lineidx_file)


def tsv_reader(tsv_file, sep='\t'):
    with exclusive_open_to_read(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str


def get_line_list(linelist_file=None, num_rows=None):
    if linelist_file is not None:
        return load_linelist_file(linelist_file)

    if num_rows is not None:
        return [i for i in range(num_rows)]


def generate_hw_file(img_file, save_file=None):
    rows = tsv_reader(img_file)

    def gen_rows():
        for i, row in tqdm(enumerate(rows)):
            row1 = [row[0]]
            img = img_from_base64(row[-1])
            height = img.shape[0]
            width = img.shape[1]
            row1.append(json.dumps([{"height": height, "width": width}]))
            yield row1

    save_file = config_save_file(img_file, save_file, '.hw.tsv')
    tsv_writer(gen_rows(), save_file)


def generate_labelmap_file(label_file, save_file=None):
    rows = tsv_reader(label_file)
    labelmap = []
    for i, row in enumerate(rows):
        labelmap.extend(set([rect['class'] for rect in json.loads(row[1])]))
    labelmap = sorted(list(set(labelmap)))

    save_file = config_save_file(label_file, save_file, '.labelmap.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(labelmap))


def extract_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)

    def gen_rows():
        for i, row in enumerate(rows):
            row1 = [row[0], row[col]]
            yield row1

    save_file = config_save_file(
        tsv_file, save_file, '.col.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def remove_column(tsv_file, col=1, save_file=None):
    rows = tsv_reader(tsv_file)

    def gen_rows():
        for i, row in enumerate(rows):
            del row[col]
            yield row

    save_file = config_save_file(
        tsv_file, save_file, '.remove.{}.tsv'.format(col))
    tsv_writer(gen_rows(), save_file)


def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected.
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([
                    any([lab[attr] for attr in ignore_attrs if attr in lab])
                    for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)


def random_drop_labels(
        label_file, drop_ratio, linelist_file=None,
        save_file=None, drop_image=False):
    # randomly drop labels by the ratio
    # if drop_image is true, can drop an image by removing all labels
    # otherwise will keep at least one label for each image to make sure
    # the number of images is equal
    rows = tsv_reader(label_file)
    line_list = get_line_list(linelist_file)
    rows_new = []
    cnt_original = 0
    cnt_new = 0
    for i, row in enumerate(rows):
        if line_list and (i not in line_list):
            row_new = [row[0], json.dumps([])]
        else:
            labels = json.loads(row[1])
            if len(labels) == 0:
                labels_new = []
            else:
                rand = np.random.random(len(labels))
                labels_new = [
                    obj for j, obj in enumerate(labels)
                    if rand[j] >= drop_ratio]
                if not drop_image and not labels_new:
                    # make sure there is at least one label
                    # if drop image is not allowed
                    labels_new = [labels[0]]
                    cnt_original += len(labels)
            cnt_new += len(labels_new)
            row_new = [row[0], json.dumps(labels_new)]
        rows_new.append(row_new)

    save_file = config_save_file(
        label_file, save_file, '.drop.{}.tsv'.format(drop_ratio))
    tsv_writer(rows_new, save_file)
    print("original labels = {}".format(cnt_original))
    print("new labels = {}".format(cnt_new))
    print("given drop_ratio = {}".format(drop_ratio))
    print("real drop_ratio = {}".format(
        float(cnt_original - cnt_new) / cnt_original))


def merge_two_label_files(label_file1, label_file2, save_file=None):
    rows1 = tsv_reader(label_file1)
    rows2 = tsv_reader(label_file2)

    rows_new = []
    for row1, row2 in zip(rows1, rows2):
        assert row1[0] == row2[0]
        labels = json.loads(row1[1]) + json.loads(row2[1])
        rows_new.append([row1[0], json.dumps(labels)])

    save_file = config_save_file(label_file1, save_file, '.merge.tsv')
    tsv_writer(rows_new, save_file)


def is_same_keys_for_files(
        tsv_file1, tsv_file2, linelist_file1=None,
        linelist_file2=None):
    # check if two files have the same keys for all rows
    tsv1 = TSVFile(tsv_file1)
    tsv2 = TSVFile(tsv_file2)
    line_list1 = get_line_list(linelist_file1, tsv1.num_rows())
    line_list2 = get_line_list(linelist_file2, tsv2.num_rows())
    assert len(line_list1) == len(line_list2)
    for idx1, idx2 in zip(line_list1, line_list2):
        row1 = tsv1.seek(idx1)
        row2 = tsv2.seek(idx2)
        if row1[0] == row2[0]:
            continue
        else:
            print("key mismatch {}-{}".format(row1[0], row2[0]))
            return False
    return True


def sort_file_based_on_keys(ref_file, tsv_file, save_file=None):
    # sort tsv_file to have the same key in each row as ref_file
    if is_same_keys_for_files(ref_file, tsv_file):
        print("file keys are the same, skip sorting")
        return tsv_file

    ref_keys = [row[0] for row in tsv_reader(ref_file)]
    all_keys = [row[0] for row in tsv_reader(tsv_file)]
    indexes = [all_keys.index(key) for key in ref_keys]
    tsv = TSVFile(tsv_file)

    def gen_rows():
        for idx in indexes:
            yield tsv.seek(idx)

    save_file = config_save_file(tsv_file, save_file, '.sorted.tsv')
    tsv_writer(gen_rows(), save_file)


def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file)
    keys = [tsv.seek(i)[0] for i in tqdm(range(len(tsv)))]
    key_to_idx = {key: i for i, key in enumerate(keys)}

    def gen_rows():
        for key in tqdm(ordered_keys):
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)


def reorder_tsv_keys_with_file(in_tsv_file, ref_tsv_file, out_tsv_file):
    ordered_keys = [row[0] for row in tsv_reader(ref_tsv_file)]
    reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file)


def convert_caption_json_to_tsv(caption_json_file, key_tsv_file, out_tsv_file):
    keys = [row[0] for row in tsv_reader(key_tsv_file)]
    rows_dict = {key: [] for key in keys}

    with open(caption_json_file, 'r') as f:
        captions = json.load(f)

    for cap in captions:
        image_id = cap['image_id']
        del cap['image_id']
        if image_id in rows_dict:
            rows_dict[image_id].append(cap)

    rows = [[key, json.dumps(rows_dict[key])] for key in keys]
    tsv_writer(rows, out_tsv_file)


def generate_caption_linelist_file(caption_tsv_file, save_file=None):
    num_captions = []
    for row in tsv_reader(caption_tsv_file):
        num_captions.append(len(json.loads(row[1])))

    cap_linelist = [
        '\t'.join([str(img_idx), str(cap_idx)])
        for img_idx in range(len(num_captions))
        for cap_idx in range(num_captions[img_idx])
    ]
    save_file = config_save_file(caption_tsv_file, save_file, '.linelist.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))


def convert_feature_format(in_tsv, out_tsv, fea_dim=None):
    # convert the old feature file format to new one
    # set fea_dim to remove spatial features if necessary.
    def gen_rows():
        for row in tqdm(tsv_reader(in_tsv)):
            key = row[0]
            feat_info = json.loads(row[1])
            num_boxes = feat_info['num_boxes']
            features = np.frombuffer(
                base64.b64decode(feat_info['features']), np.float32
                ).reshape(num_boxes, -1)
            if fea_dim:
                feat_info_new = [
                    {'feature': base64.b64encode(
                        features[i][:fea_dim]).decode('utf-8')}
                    for i in range(num_boxes)]
            else:
                feat_info_new = [
                    {'feature': base64.b64encode(
                        features[i]).decode('utf-8')}
                    for i in range(num_boxes)]
            yield [key, json.dumps(feat_info_new)]
    tsv_writer(gen_rows(), out_tsv)


def convert_feature_format2(in_tsv, out_tsv, fea_dim=None):
    # new format from Pengchuan
    def gen_rows():
        for row in tqdm(tsv_reader(in_tsv)):
            key = row[0]
            num_boxes = int(row[1])
            features = np.frombuffer(
                base64.b64decode(row[2]), np.float32).reshape(num_boxes, -1)
            if fea_dim:
                feat_info = [
                    {'feature': base64.b64encode(
                        features[i][:fea_dim]).decode('utf-8')}
                    for i in range(num_boxes)]
            else:
                feat_info = [
                    {'feature': base64.b64encode(
                        features[i]).decode('utf-8')}
                    for i in range(num_boxes)]
            yield [key, json.dumps(feat_info)]
    tsv_writer(gen_rows(), out_tsv)


def merge_label_fields(in_tsv1, in_tsv2, out_tsv):
    # merge the label fields for each box
    def gen_rows():
        for row1, row2 in tqdm(zip(tsv_reader(in_tsv1), tsv_reader(in_tsv2))):
            assert row1[0] == row2[0]
            label_info1 = json.loads(row1[1])
            label_info2 = json.loads(row2[1])
            assert len(label_info1) == len(label_info2)
            for lab1, lab2 in zip(label_info1, label_info2):
                lab1.update(lab2)
            yield [row1[0], json.dumps(label_info1)]
    tsv_writer(gen_rows(), out_tsv)


def remove_label_fields(in_tsv, out_tsv, remove_fields):
    if type(remove_fields) == str:
        remove_fields = [remove_fields]
    assert type(remove_fields) == list

    def gen_rows():
        for row in tqdm(tsv_reader(in_tsv)):
            label_info = json.loads(row[1])
            for lab in label_info:
                for field in remove_fields:
                    if field in lab:
                        del lab[field]
            yield [row[0], json.dumps(label_info)]
    tsv_writer(gen_rows(), out_tsv)


def random_permute_label_file(in_tsv, out_tsv):
    # take a label file as input and randomly match image
    # with the label from a different image
    tsv = TSVFile(in_tsv)
    random_index = np.random.permutation(tsv.num_rows())

    def gen_rows():
        for idx, rand_idx in enumerate(random_index):
            key = tsv.seek(idx)[0]
            labels = tsv.seek(rand_idx)[1]
            yield [key, labels]
    tsv_writer(gen_rows(), out_tsv)
    # save the random index for reference
    save_file = op.splitext(out_tsv)[0] + '.random_index.tsv'
    with open(save_file, 'w') as f:
        f.write('\n'.join([str(idx) for idx in random_index]))


def create_mini_yaml_with_linelist(in_yaml, num_files):
    # create linelist files to split a yaml into multiple ones
    # useful for inference on large-scale dataset
    data_cfg = load_from_yaml_file(in_yaml)
    data_dir = op.dirname(in_yaml)
    split_name = op.basename(in_yaml).split('.')[0]
    hw_file = op.join(data_dir, data_cfg['hw'])
    num_rows = TSVFile(hw_file).num_rows()
    rows_per_file = math.ceil(num_rows / num_files)
    for idx in range(num_files):
        start_idx = idx * rows_per_file
        end_idx = min(start_idx + rows_per_file, num_rows)
        linelist = [str(i) for i in range(start_idx, end_idx)]
        linelist_file = op.join(
            data_dir, split_name + '.linelist_{}.tsv'.format(idx))
        print("create linelist file: " + linelist_file)
        with open(linelist_file, 'w') as f:
            f.write('\n'.join(linelist))
        data_cfg['linelist'] = op.basename(linelist_file)
        out_yaml = op.splitext(in_yaml)[0] + '_{}.yaml'.format(idx)
        write_to_yaml_file(data_cfg, out_yaml)
        print("create yaml file: " + out_yaml)


def mapping_labels(in_tsv, out_tsv, label_mapping_dict):
    def gen_rows():
        for row in tsv_reader(in_tsv):
            label_info = json.loads(row[1])
            for lab in label_info:
                if lab['class'] in label_mapping_dict:
                    lab['class'] = label_mapping_dict[lab['class']]
            yield [row[0], json.dumps(label_info)]
    tsv_writer(gen_rows(), out_tsv)


def select_rows_in_linelist(in_tsv, out_tsv, linelist_file):
    tsv = TSVFile(in_tsv)
    line_list = load_linelist_file(linelist_file)

    def gen_rows():
        for idx in line_list:
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv)


def generate_full_region_label_file(
        hw_tsv, out_tsv, class_name=None):
    # given a height/width file, generate a label file
    def gen_rows():
        for row in tsv_reader(hw_tsv):
            try:
                data = json.loads(row[1])
                assert type(data) in (list, dict)
                if type(data) == list:
                    height, width = data[0]['height'], data[0]['width']
                else:
                    height, width = data['height'], data['width']
            except ValueError:
                hw_str = row[1].split(' ')
                height, width = int(hw_str[0]), int(hw_str[1])
            label = {'rect': [0, 0, width, height]}
            if class_name:
                label.update({'class': class_name})
            yield [row[0], json.dumps([label])]
    tsv_writer(gen_rows(), out_tsv)

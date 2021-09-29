import numpy as np
import os
import stat
import grp
import re
import sys

from argparse import ArgumentParser
from pathlib import Path

try:
    from datapipeline.load_tiff import tiffy
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        'Import statement `from datapipeline.load_tiff import tiffy`'
        ' failed. Make sure you run this script in the Cai Lab shared '
        'python environment where the datapipeline scripts are available.'
    )

# Check presence of Hyb* folders - only numbers, raise warning if
# non-consecutive; should have read + execute access
# Within each Hyb* folder, check presence of MMStack_Pos(\d+).ome.tif images.
# Should all be readable, same filesize.
# Finally, slow step: try to open each one with tiffy.load() and report errors
# or any that have anomalous shape.

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', help='The path to the root folder of'
                        ' the experiment to check. This is the folder that '
                        'contains the `HybCycle_*` folders.')
    parser.add_argument('--fast', action='store_true', help='If supplied, only '
                        'perform the fast steps of file validation; do not try to '
                        'open every TIFF, only check existence and permissions of '
                        'files.')

    args = parser.parse_args()

    return args

lab_grnam = 'hpc_CaiLab'
lab_gid = grp.getgrnam(lab_grnam).gr_gid
relevant_folders = ['initial_background', 'final_background', 'segmentation',
                    'Labeled_Images', 'Labeled_Images_Cytoplasm']

def test_open_tiff(fname):
    im = tiffy.load(fname)

    return im

def test_tiff_dtype(im):
    if im.dtype != np.uint16:
        raise TypeError(f'Image datatype is {im.dtype}, should be'
                        'np.uint16.')

def test_hyb_cycle_name(folder):
    if isinstance(folder, Path):
        folder = folder.name

    full_hyb_folder_re = re.compile('^HybCycle_(\d+)$')

    full_match = full_hyb_folder_re.match(folder)

    if full_match is None:
        raise ValueError(f'Invalid folder name `{folder}`. '
                         'Any folder containing `Hyb` should be '
                         'of the form HybCycle_N, where N is a number.'
                         )
    return full_match.group(1)

def test_tiff_stack_name(fname):
    if isinstance(fname, Path):
        fname = fname.name

    tiff_stack_re = re.compile('^MMStack_Pos(\d+)\.ome\.tif$')

    match = tiff_stack_re.match(fname)

    if match is None:
        raise ValueError(f'Invalid TIFF stack name `{fname}`. '
                         'Each TIFF stack should be of the form '
                         'MMStack_PosN.ome.tif, where N is a number.'
                         )
    return match.group(1)


def test_gid(path_stat):
    if path_stat.st_gid != lab_gid:
        raise PermissionError(f'Group ID does not match that of '
                              f'{lab_grnam}, {lab_gid}. Need to `chgrp` this file.')

def test_user_perms(path_stat):
    mode = path_stat.st_mode
    user_perms = (mode & stat.S_IRWXU) >> 6

    if stat.S_ISDIR(mode):
        req_user = 7
    else:
        req_user = 6

    if user_perms < req_user:
        raise PermissionError(f'File requires owner permission mode {req_user}')

def test_group_perms(path_stat):
    mode = path_stat.st_mode
    group_perms = (mode & stat.S_IRWXG) >> 3

    if stat.S_ISDIR(mode):
        req_group = 5
    else:
        req_group = 4

    if group_perms < req_group:
        raise PermissionError(f'File requires group permission mode {req_group}')


def test_stat(path):
    path = Path(path)
    errors = []
    size = -1

    try:
        path_stat = path.stat()
        size = path_stat.st_size
    except (OSError, PermissionError) as e:
        errors.append(e)

    for test in (test_gid, test_user_perms, test_group_perms):
        try:
            test(path_stat)
        except PermissionError as e:
            errors.append(e)

    return errors, size

def find_relevant_folders(root):
    root = Path(root)

    subfolders = [d for d in root.iterdir() if d.is_dir()]

    extra_folders = []
    hyb_folder_info = []

    for f in subfolders:
        stat_errors, _ = test_stat(f)
        entry = dict(path=f, name=f.name, number=-1, contents=[], errors=stat_errors)

        if 'Hyb' in f.name:
            try:
                hyb_num = test_hyb_cycle_name(f.name)
                entry['number'] = int(hyb_num)
            except ValueError as e:
                entry['errors'].append(e)

            hyb_folder_info.append(entry)
        elif f.name in relevant_folders:
            hyb_folder_info.append(entry)
        else:
            extra_folders.append(f.name)

    return hyb_folder_info, extra_folders


def test_image_folder_contents(folder):
    folder = Path(folder)
    contents = list(folder.iterdir())

    folder_contents_info = []

    for c in contents:
        stat_errors, size = test_stat(c)
        entry = dict(path=c, name=c.name, number=None, size=size, errors=stat_errors)

        try:
            pos_number = test_tiff_stack_name(c)
            entry['number'] = int(pos_number)
        except ValueError as e:
            entry['errors'].append(e)

        folder_contents_info.append(entry)

    return folder_contents_info

def test_image_opening(entry):
    entry['shape'] = None
    entry['opening_errors'] = []
    try:
        im = test_open_tiff(entry['path'])
        entry['shape'] = im.shape
    except Exception as e:
        entry['opening_errors'].append(e)

    try:
        test_tiff_dtype(im)
    except TypeError as e:
        entry['opening_errors'].append(e)

    del im

    return entry

def check_all_images(results, hyb_nums=None, pos_nums=None):

    for hyb_entry in results:
        if len(hyb_entry['errors']) > 0:
            continue
        if (hyb_nums is not None
            and hyb_entry['number'] not in hyb_nums):
            continue

        for im_entry in hyb_entry['contents']:
            if len(im_entry['errors']) > 0:
                continue
            if (pos_nums is not None
                and im_entry['number'] not in pos_nums):
                continue

            im_name = im_entry['name']
            im_entry = test_image_opening(im_entry)

            im_open_errors = im_entry['opening_errors']
            print(f'{im_name}: {len(im_open_errors)} errors opening TIFF')
            for e in im_open_errors:
                print(f'        ', str(e))

    return results

def check_tree(root):
    root = Path(root)

    hyb_folder_info, extra_folders = find_relevant_folders(root)

    for folder_entry in hyb_folder_info:
        folder_entry['contents'] = test_image_folder_contents(folder_entry['path'])

    return hyb_folder_info, extra_folders


def sort_key(d):
    n = d['number']
    if n is None:
        return -1
    else:
        return n

def print_errors(results, extra_folders):
    results = sorted(results, key=sort_key)

    for extra in extra_folders:
        print(f'Note: extra folder {extra}')

    for folder_entry in results:
        folder_errors = folder_entry['errors']
        folder_contents_errors = sorted([
            c for c in folder_entry['contents']
            if len(c['errors']) > 0
        ], key=sort_key)

        folder_name = folder_entry['name']
        folder_number = folder_entry['number']

        if len(folder_errors) == 0 and len(folder_contents_errors) == 0:
            print(f'{folder_name}: No problems.')
            continue

        print(f'{folder_name}')

        if len(folder_errors) > 0:
            print(f'{len(folder_errors)} folder errors.')
            for e in folder_errors:
                print('    ', str(e))

        if len(folder_contents_errors) > 0:
            for c in folder_contents_errors:
                im_name = c['name']
                im_number = c['number']
                im_errors = c['errors']
                print(f'    {folder_name} / {im_name} had {len(im_errors)} errors:')
                for e in im_errors:
                    print('        ', str(e))


def main(args):
    print_errors(*check_tree(args.root))

if __name__ == '__main__':
    args = parse_args()
    main(args)

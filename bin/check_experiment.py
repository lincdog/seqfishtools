import numpy as np
import os
import stat
import grp
import re
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from pathlib import Path
from copy import copy

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
    parser.add_argument('--output', default=sys.stdout,
                        help='The filename to print the '
                        'results report to. By default goes to stdout.')
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
    if im is None:
        return

    assert im.dtype == np.uint16, \
        f'Image datatype is {im.dtype}, should be np.uint16.'


def test_hyb_cycle_name(folder):
    if isinstance(folder, Path):
        folder = folder.name

    full_hyb_folder_re = re.compile('^HybCycle_(\d+)$')

    full_match = full_hyb_folder_re.match(folder)

    assert full_match is not None, \
        f'Invalid folder name `{folder}`. Any folder containing `Hyb` should be ' \
        f'of the form HybCycle_N, where N is a number.'

    return full_match.group(1)


def test_tiff_stack_name(fname):
    if isinstance(fname, Path):
        fname = fname.name

    tiff_stack_re = re.compile('^MMStack_Pos(\d+)\.ome\.tif$')

    match = tiff_stack_re.match(fname)

    assert match is not None, \
        f'Invalid TIFF stack name `{fname}`. Each TIFF stack should be of the form ' \
        f'MMStack_PosN.ome.tif, where N is a number.'

    return match.group(1)


def test_gid(path_stat):
    assert path_stat.st_gid == lab_gid, \
        f'Group ID does not match that of ' \
        f'{lab_grnam}, {lab_gid}. Need to `chgrp` this file.'


def test_user_perms(path_stat):
    mode = path_stat.st_mode
    user_perms = (mode & stat.S_IRWXU) >> 6

    if stat.S_ISDIR(mode):
        req_user = (7,)
    else:
        req_user = (6, 7)

    assert user_perms in req_user, \
        f'File requires owner permission mode {req_user}'


def test_group_perms(path_stat):
    mode = path_stat.st_mode
    group_perms = (mode & stat.S_IRWXG) >> 3

    if stat.S_ISDIR(mode):
        req_group = (5, 7)
    else:
        req_group = (4, 5, 6, 7)

    assert group_perms in req_group, \
        f'File requires group permission mode {req_group}'


def test_stat(path):
    path = Path(path)
    errors = []
    size = -1

    try:
        path_stat = path.stat()
        size = path_stat.st_size
    except AssertionError as e:
        errors.append(e)

    for test in (test_gid, test_user_perms, test_group_perms):
        try:
            test(path_stat)
        except AssertionError as e:
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
            except AssertionError as e:
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
        except AssertionError as e:
            entry['errors'].append(e)

        folder_contents_info.append(entry)

    return folder_contents_info


def test_image_opening(entry):
    updated_entry = copy(entry)
    updated_entry['shape'] = None

    updated_entry['opening_errors'] = []
    im = None

    try:
        im = test_open_tiff(updated_entry['path'])
        updated_entry['shape'] = im.shape
    except Exception as e:
        updated_entry['opening_errors'].append(e)

    try:
        test_tiff_dtype(im)
    except AssertionError as e:
        updated_entry['opening_errors'].append(e)

    del im

    return updated_entry


def check_all_images(results, hyb_nums=None, pos_nums=None):
    to_check = []

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

            im_entry_copy = copy(im_entry)
            im_entry_copy['hyb_name'] = hyb_entry['name']
            im_entry_copy['hyb_number'] = hyb_entry['number']
            to_check.append(im_entry_copy)

    opening_errors = []
    has_opening_errors = False

    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = []
        for im_entry in to_check:
            futures.append(exe.submit(test_image_opening, im_entry))
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 5 == 0:
                print(f'done with {done}/{len(futures)} images')

            result = fut.result(1)
            if len(result['opening_errors']) > 0:
                has_opening_errors = True
                opening_errors.append(result)

    return opening_errors, has_opening_errors


def check_tree(root):
    root = Path(root)

    has_errors = False

    hyb_folder_info, extra_folders = find_relevant_folders(root)

    for folder_entry in hyb_folder_info:
        if len(folder_entry['errors']) > 0:
            has_errors = True

        folder_entry['contents'] = test_image_folder_contents(folder_entry['path'])

        if any([len(c['errors']) > 0 for c in folder_entry['contents']]):
            has_errors = True

    return hyb_folder_info, extra_folders, has_errors


def global_checks(results, image_results=None):
    pass


def sort_key(d):
    n = d['number']
    if n is None:
        return -1
    else:
        return n


def print_errors(results, extra_folders, output_file):
    print('File Structure and Permissions Errors', file=output_file)
    print('---------------------------------------', file=output_file)
    results = sorted(results, key=sort_key)

    for extra in extra_folders:
        print(f'Note: extra folder `{extra}`', end='\n\n', file=output_file)

    for folder_entry in results:
        folder_errors = folder_entry['errors']
        folder_contents_errors = sorted([
            c for c in folder_entry['contents']
            if len(c['errors']) > 0
        ], key=sort_key)

        folder_name = folder_entry['name']
        folder_number = folder_entry['number']

        if len(folder_errors) == 0 and len(folder_contents_errors) == 0:
            print(f'`{folder_name}`: No problems.', end='\n\n', file=output_file)
            continue

        if len(folder_errors) > 0:
            print(f'`{folder_name}`: {len(folder_errors)} folder errors.', file=output_file)
            for e in folder_errors:
                print('    ', str(e), file=output_file)

        if len(folder_contents_errors) > 0:
            for c in folder_contents_errors:
                im_name = c['name']
                im_number = c['number']
                im_errors = c['errors']
                print(f'    {folder_name} / {im_name} had {len(im_errors)} errors:',
                      file=output_file)
                for e in im_errors:
                    print('        ', str(e), file=output_file)
        print('\n\n', file=output_file)


def print_opening_errors(opening_results, output_file):
    print('----------------------------', file=output_file)
    print('Image Opening Errors', file=output_file)

    for c in opening_results:
        im_name = c['name']
        hyb_name = c['hyb_name']
        im_number = c['number']
        im_errors = c['opening_errors']
        print(f'    {hyb_name} / {im_name} had {len(im_errors)} errors:',
              file=output_file)
        for e in im_errors:
            print('        ', str(e), file=output_file)


def main(args):
    has_errors = False
    has_opening_errors = False

    if args.output != sys.stdout:
        output_parent = Path(args.output).parent
        output_parent.mkdir(parents=True, exist_ok=True, mode=0o2770)

    with open(args.output, 'w') as output_file:
        results, extras, has_errors = check_tree(args.dataset)

        print_errors(results, extras, output_file)

        if not args.fast:
            opening_results, has_opening_errors = check_all_images(results)
            print_opening_errors(opening_results, output_file)

    return int(has_errors or has_opening_errors)

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

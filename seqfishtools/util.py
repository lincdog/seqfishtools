import re
import os
import json
import string
import numpy as np
import pandas as pd
import jmespath
import xmlschema
import tifffile as tif

from lxml import etree
from pathlib import Path, PurePath
from PIL import Image, ImageSequence, UnidentifiedImageError

# Namespace prefixes for convenience when working with xpath
ns = {
    'OME': 'http://www.openmicroscopy.org/Schemas/OME/2015-01',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}
# Mapping of schema source URL to local file path
schemas = {
    'http://www.openmicroscopy.org/Schemas/OME/2015-01':
    Path(__file__).parent / 'xml/ome.xsd'
}


def _convert_types(data):
    data = dict(data)

    def _convert(i):
        r = i
        try:
            if '.' in i:
                r = float(i)
            else:
                r = int(i)
        except (TypeError, ValueError):
            pass
        return r

    for k, v in data.items():
        if not isinstance(v, str):
            continue

        data[k] = _convert(v)

    return data


class OMEImageLoader:

    def __init__(
            self,
            fname,
            schema=None,

    ):
        self.metadata_file = None
        self.raw_metadata = None
        self.metadata_etree = None
        self.metadata = None
        self.schema = None
        self.master_tiff = None
        self.root = None

        given_metadata = tif.TiffFile(fname).ome_metadata

        if not given_metadata:
            raise ValueError('The supplied file has no OME metadata readable by tifffile.')

        fpath = Path(fname)

        given_tree = etree.XML(given_metadata.encode())
        root_namespace = given_tree.nsmap.get(None)

        if root_namespace in schemas:
            schema_location = schemas[root_namespace]
        else:
            schema_location = given_tree.xpath(
                '@xsi:schemaLocation', namespaces=ns).split(' ')[-1]

        try:
            self.schema = xmlschema.XMLSchema(schema_location)
        except:
            self.schema = None

        metadata_ref = given_tree.xpath('OME:*//@MetadataFile', namespaces=ns)

        if len(metadata_ref) > 0:
            metadata_ref[0] = str(metadata_ref[0])
            metadata_file = Path(metadata_ref[0])

            if not metadata_file.is_absolute():
                # Make it relative to the supplied filename
                metadata_file = fpath.with_name(metadata_ref[0])

            if not metadata_file.exists():
                raise FileNotFoundError(
                    f'Supplied file {fname} referenced metadata file {metadata_ref[0]},'
                    f' but this file was not found.'
                )
        else:
            metadata_file = fpath

        self.metadata_file = metadata_file.resolve()
        self.master_tiff = tif.TiffFile(metadata_file)
        self.root = metadata_file.parent
        self.raw_metadata = self.master_tiff.ome_metadata.encode()

        self.metadata_etree = etree.XML(self.raw_metadata)

        result = dict()

        for im in self.metadata_etree.xpath('OME:Image', namespaces=ns):
            im_name = im.attrib['Name']

            im_info = im.xpath(
                'OME:Pixels',
                namespaces=ns
            )[0].attrib

            im_channels = [
                _convert_types(chan.attrib)
                for chan in im.xpath(
                    'OME:Pixels/OME:Channel',
                    namespaces=ns)
            ]

            im_planes = [
                _convert_types(plane.attrib)
                for plane in im.xpath('OME:Pixels/OME:Plane', namespaces=ns)
            ]

            im_info = dict(im_info)
            im_info['channels'] = im_channels
            im_info['planes'] = im_planes
            result[im_name] = im_info

        self.metadata = result

    def get_metadata(self, fname):
        fname = Path(fname).resolve()

        if not self.root.samefile(fname.parent):
            raise ValueError(f'Target file {fname} is not in the same directory '
                             f'as instance metadata file {self.metadata_file}.')

        stem = fname.name.removesuffix('.ome.tif')

        return self.metadata[stem]

    def imread(self, fname):
        plane_metadata = self.get_metadata(fname)['planes']

        pil_im = pil_imopen(fname)

        im_arr = pil_frames_to_ndarray(
            pil_im,
            plane_metadata,
            dtype=np.uint16,
            channel_key='TheC',
            slice_key='TheZ'
        )

        return im_arr


def make_align_name(position, hyb):
    return f'alignment/pos_{position}_hyb_{hyb}_offset'


def absolute(path):
    return str(Path(path).absolute().resolve())


def for_each_pos_each_hyb(exp_tab, fmt_string):
    df = pd.read_csv(exp_tab)
    pos_hybs = df[['position', 'hyb']].values

    fmt_string = str(fmt_string)

    return [fmt_string.format(position=p, hyb=h) for p, h in pos_hybs]


_MM_TIFF_TAG_NUMBER = 51123


def pil_imopen(fname, metadata=False):
    im = Image.open(fname)

    if metadata:
        return im, pil_getmetadata(im)
    else:
        return im


def pil_imread(
    fname,
    metadata=False,
    swapaxes=False,
    ensure_4d=True,
    backup=tif.imread,
    **kwargs
):
    md = None

    import warnings
    warnings.simplefilter('ignore', UserWarning)

    pil_succeeded = False

    try:
        im = pil_imopen(fname)
        md = pil_getmetadata(im)
        imarr = pil_frames_to_ndarray(im, md)
        pil_succeeded = True
    except (ValueError, UnidentifiedImageError, AssertionError) as e:

        if callable(backup):
            imarr = backup(fname, **kwargs)
        else:
            raise e

    if ensure_4d and imarr.ndim == 3:
        # assumes 1 Z
        imarr = imarr[:, None, :]

    # Updated 8/23/21: We only want to swap axes if the PIL metadata was present.
    # Once we open this way and save, tif.imread() should always read it in the same way,
    # which is what we want. Swapping at this point means each time we open a saved image
    # the axes will be swapped over and over.
    if pil_succeeded and swapaxes and imarr.ndim == 4:
        imarr = imarr.swapaxes(0, 1)

    if metadata and md:
        return imarr, md
    else:
        return imarr


def pil_getmetadata(im, relevant_keys=None):
    """
    pil_getmetadata
    ---------------
    Given a PIL image sequence im, retrieve the metadata associated
    with each frame in the sequence. Only keep metadata keys specified
    in `relevant_keys` - which will default to ones that we need such as
    channel, slice information. There are many metadata keys which are
    useless / do not change frame to frame.
    Returns: List of dicts in order of frame index.
    """

    if str(relevant_keys).lower() == 'all':
        relevant_keys = None

    elif not isinstance(relevant_keys, list):

        relevant_keys = [
            'Andor sCMOS Camera-Exposure',  # Exposure time (ms)
            'Channel',                      # Channel name (wavelength)
            'ChannelIndex',                 # Channel index (number)
            'Frame',                        # Time slice (usually not used)
            'FrameIndex',                   # Time slice index (usually not used)
            'PixelSizeUm',                  # XY pixel size in microns
            'Position',                     # Position
            'PositionIndex',                # Position index (MMStack_PosX)
            'PositionName',                 # Position name
            'Slice',                        # Z slice
            'SliceIndex'                    # Z slice index (same as Slice)
        ]

    frame_metadata = []

    for frame in ImageSequence.Iterator(im):

        if _MM_TIFF_TAG_NUMBER in frame.tag_v2:
            jsstr = frame.tag_v2[_MM_TIFF_TAG_NUMBER]
            jsdict = json.loads(jsstr)

            if relevant_keys:
                # Only keep the relevant keys
                rel_dict = {
                    k: jsdict.get(k)
                    for k in relevant_keys
                }
            else:
                rel_dict = jsdict

            frame_metadata.append(rel_dict)

    return frame_metadata


def pil2numpy(im, dtype=np.uint16):

    return np.frombuffer(im.tobytes(), dtype=dtype).reshape(im.size)


def pil_frames_to_ndarray(
        im,
        metadata,
        dtype=np.uint16,
        channel_key='ChannelIndex',
        slice_key='SliceIndex'
):
    """
    pil_frames_to_ndarray
    -----------------
    Given a PIL image sequence, return a Numpy array that is correctly
    ordered and shaped as (n_channels, n_slices, ...) so that we can
    process it in a consistent way.
    To do this, we look at the ChannelIndex and SliceIndex of each frame
    in the stack, and insert them one by one into the correct position
    of a 4D numpy array.
    """
    #metadata = pil_getmetadata(im)

    if not metadata:
        raise ValueError('Supplied image lacks metadata used for '
            'forming the correct image shape. Was the image not '
            'taken from ImageJ/MicroManager?')

    # Gives a list of ChannelIndex for each frame
    cinds = jmespath.search(f'[].{channel_key}', metadata)
    # Gives a list of SliceIndex for each frame
    zinds = jmespath.search(f'[].{slice_key}', metadata)

    if (len(cinds) != len(zinds)
        or any([c is None for c in cinds])
        or any([z is None for z in zinds])
    ):
        raise ValueError('SuppliedImage lacks `ChannelIndex` or '
                         '`SliceIndex` metadata required to form '
                         'properly shaped numpy array. Was the image not '
                         'taken directly from ImageJ/MicroManager?')

    ncs = max(cinds) + 1
    nzs = max(zinds) + 1

    total_frames = ncs * nzs
    assert total_frames == im.n_frames, 'wrong shape'

    # Concatenate the channel and slice count to the XY shape in im.size
    new_shape = (ncs, nzs) + im.size

    # Make an empty ndarray of the proper shape and dtype
    npoutput = np.empty(new_shape, dtype=dtype)

    # Loop in a nested fashion over channel first then Z slice
    for c in range(ncs):
        for z in range(nzs):

            # Find the frame whose ChannelIndex and SliceIndex
            # match the current c and z values
            entry = jmespath.search(
                f'[?{channel_key}==`{c}` && {slice_key}==`{z}`]', metadata)[0]

            # Find the *index* of the matching frame so that we can insert it
            ind = metadata.index(entry)

            # Select the matching frame
            im.seek(ind)

            # Copy the frame into the correct c and z position in the numpy array
            npoutput[c, z] = pil2numpy(im)

    return npoutput


def fmt2regex(fmt, delim=os.path.sep):
    """
    fmt2regex:
    convert a curly-brace format string with named fields
    into a regex that captures those fields as named groups,
    Returns:
    * reg: compiled regular expression to capture format fields as named groups
    * globstr: equivalent glob string (with * wildcards for each field) that can
        be used to find potential files that will be analyzed with reg.
    """
    sf = string.Formatter()

    regex = []
    globstr = []
    keys = set()

    numkey = 0

    fmt = str(fmt).rstrip(delim)

    if delim:
        parts = fmt.split(delim)
    else:
        delim = ''
        parts = [fmt]

    re_delim = re.escape(delim)

    for part in parts:
        part_regex = ''
        part_glob = ''

        for a in sf.parse(part):
            r = re.escape(a[0])

            newglob = a[0]
            if a[1]:
                newglob = newglob + '*'
            part_glob += newglob

            if a[1] is not None:
                k = re.escape(a[1])

                if len(k) == 0:
                    k = f'k{numkey}'
                    numkey += 1

                if k in keys:
                    r = r + f'(?P={k})'
                else:
                    r = r + f'(?P<{k}>[^{re_delim}]+)'

                keys.add(k)

            part_regex += r

        globstr.append(part_glob)
        regex.append(part_regex)

    reg = re.compile('^'+re_delim.join(regex))
    globstr = delim.join(globstr)

    return reg, globstr


def find_matching_files(base, fmt, paths=None):
    """
    findAllMatchingFiles: Starting within a base directory,
    find all files that match format `fmt` with named fields.
    Returns:
    * files: list of filenames, including `base`, that match fmt
    * keys: Dict of lists, where the keys are each named key from fmt,
        and the lists contain the value for each field of each file in `files`,
        in the same order as `files`.
    """

    reg, globstr = fmt2regex(fmt)

    base = PurePath(base)

    files = []
    mtimes = []
    keys = {}

    if paths is None:
        paths = Path(base).glob(globstr)
    else:
        paths = [Path(p) for p in paths]

    for f in paths:
        m = reg.match(str(f.relative_to(base)))

        if m:
            try:
                mtimes.append(os.stat(f).st_mtime)
            except (PermissionError, OSError):
                mtimes.append(-1)

            files.append(f)

            for k, v in m.groupdict().items():
                if k not in keys.keys():
                    keys[k] = []

                keys[k].append(v)

    return files, keys, mtimes


def sort_as_num_or_str(coll, numtype=int, return_string=True):

    try:
        sorted_coll = sorted([numtype(i) for i in coll])
    except ValueError:
        sorted_coll = sorted(coll)

    if return_string:
        return [str(i) for i in sorted_coll]
    else:
        return sorted_coll


def get_stage_positions(filename, px_size=0.111, to_df=False):
    """
    get_stage_positions:
    Parse a MicroManager position list to return the X,Y,Z
    positions and names of each position.
    """

    with open(filename) as f:
        content = json.load(f)

    positions = []

    for i, pos in enumerate(content['POSITIONS']):

        z_name = pos['DEFAULT_Z_STAGE']
        xy_name = pos['DEFAULT_XY_STAGE']

        gridcol = pos.get('GRID_COL', -1)
        gridrow = pos.get('GRID_ROW', -1)

        # conversion from position file label convention
        # to output image convention
        if gridcol != -1 and gridrow != -1:
            # pos_1d = ((gridrow % 2) * -1)
            # pos['LABEL'] = f'Pos{(gridcol+1)*(gridrow)}'
            pos['LABEL'] = f'Pos{i}'

        posinfo = {
            'label': pos['LABEL'],
            'gridrow': gridrow,
            'gridcol': gridcol
        }

        for dev in pos['DEVICES']:

            if dev['DEVICE'] == z_name:
                posinfo['z'] = dev['X']

            if dev['DEVICE'] == xy_name:
                posinfo['x'] = dev['X']
                posinfo['xpx'] = round(dev['X'] / px_size)
                posinfo['y'] = dev['Y']
                posinfo['ypx'] = round(dev['Y'] / px_size)

        positions.append(posinfo)

    if to_df:
        return pd.DataFrame(positions)
    else:
        return positions


def get_stage_positions_2(filename, px_size=0.111, to_df=False):
    """
    An alternative implementation of get_stage_positions with jmespath
    """

    with open(filename, 'r') as f:
        content = json.load(f)

    positions = jmespath.search(
        'POSITIONS[].{label: LABEL, '
        'gridrow: GRID_ROW, '
        'gridcol: GRID_COL, '
        'x: sum(DEVICES[?DEVICE == `XYStage`].X), '
        'y: sum(DEVICES[?DEVICE == `XYStage`].Y), '
        'z: sum(DEVICES[?DEVICE == `Adaptive Focus Control Offset`].X)}',
        content
    )

    for pos in positions:
        pos['xpx'] = round(pos['x'] / px_size)
        pos['ypx'] = round(pos['y'] / px_size)

    if to_df:
        return pd.DataFrame(positions)
    else:
        return positions
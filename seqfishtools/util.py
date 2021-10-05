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

hash_format_classes = dict()


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
        self.given_file = None
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
        self.given_file = fpath

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

    def _resolve_fname(self, fname=None):
        if not fname:
            fname = self.given_file

        fname = Path(fname)

        if len(fname.parts) == 1:
            fname = self.root / fname

        fname = fname.resolve()

        return fname

    def get_metadata(self, fname=None):

        fname = self._resolve_fname(fname)

        if not self.root.samefile(fname.parent):
            raise ValueError(f'Target file {fname} is not in the same directory '
                             f'as instance metadata file {self.metadata_file}.')

        stem = fname.name.removesuffix('.ome.tif')

        return self.metadata[stem]

    def imread(self, fname=None):

        fname = self._resolve_fname(fname)

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


class ImHashFormat:
    """
    Base class for generating 2D frame hash-based image metadata
    """
    @staticmethod
    def hash_frame(frame):
        raise NotImplementedError

    @classmethod
    def encode(cls, inds):
        raise NotImplementedError

    @classmethod
    def decode(cls, val):
        raise NotImplementedError


class ImHashV1(ImHashFormat):
    prefix = '__ImHashV1:'
    delim = ','

    @staticmethod
    def hash_frame(frame):
        return hash(frame.tobytes())

    @classmethod
    def encode(cls, inds):
        joined = cls.delim.join([str(i) for i in inds])

        return cls.prefix + joined

    @classmethod
    def decode(cls, val):
        assert val.startswith(cls.prefix), f'Invalid prefix, should be {cls.prefix}'

        val_stripped = val.removeprefix(cls.prefix)

        return tuple([int(s) for s in val_stripped.split(cls.delim)])

class ImHashSparse(ImHashV1):
    prefix = '__ImHashSparse:'

    @staticmethod
    def hash_frame(frame):
        return hash(frame.ravel()[::200].tobytes())

def hash_nd_image(im, format_class=ImHashV1):
    """
    Given an ndarray of any shape, generate a dict where the keys are hash values
    of its 2D slices, and the values are a tuple of coordinate indices that correspond
    to that 2D slice. This is assuming the last 2 dimensions of the array are Y and X,
    and that the array is a (n-2)d compendium of these 2D images.
    """

    hashes = dict()

    for inds in np.ndindex(im.shape[:-2]):
        slice_hash = format_class.hash_frame(im[inds])
        hashes[slice_hash] = inds

    return hashes


def encode_nd_hashes(hashes, format_class=ImHashV1):
    """
    Given a dict of the form returned by hash_nd_image, returns a string
    formatted dict according to the format_class given. Converts the keys
    (hash values) to string, and uses the format_class.encode() function on
    the tuple of coordinates for each item.
    """
    hashes_formatted = dict()

    for k_int, inds in hashes.items():
        k_str = str(k_int)

        formatted_inds = format_class.encode(inds)

        hashes_formatted[k_str] = formatted_inds

    return hashes_formatted


def decode_nd_hashes(hashes_formatted, format_class=ImHashV1):
    """
    Reverse of encode_nd_hashes: given a string-formatted dict of hash, coordinate
    indices, uses the format_class.decode() method to convert back to a tuple of
    coordinate indices.
    """

    hashes = dict()
    errors = dict()

    for k, v in hashes_formatted.items():
        try:
            assert isinstance(v, str)
            k_int = int(k)
            val_decoded = format_class.decode(v)
        except (AssertionError, ValueError) as e:
            errors[k] = e
        else:
            hashes[k_int] = val_decoded

    if len(errors) == len(hashes_formatted):
        raise ValueError(f'Unable to decode metadata values using {format_class.__name__}.'
                         ' Are you using the correct format class for this image?')

    return hashes


def _get_image_description(image):
    """
    Given a PIL.Image.Image or tifffile.TiffFile instance, returns the appropriate
    metadata dict (from the 'ImageDescription' tag) that holds the hash: coord pairs
    when saved using hash_imwrite.
    """
    if isinstance(image, Image.Image):
        init_pos = image.tell()
        image.seek(0)
        image_description = image.tag_v2.named().get('ImageDescription')

        try:
            raw = json.loads(image_description)
        except json.decoder.JSONDecodeError:
            raise ValueError('Supplied image ImageDescription tag does not contain'
                             ' valid JSON data.')
        finally:
            image.seek(init_pos)

    elif isinstance(image, tif.TiffFile):
        raw = image.shaped_metadata[0]
    else:
        raise TypeError('Unsupported argument to get_image_metadata; must be '
                        'PIL.Image or tifffile.TiffFile.')

    if not raw:
        raise ValueError('Empty ImageDescription metadata for image')

    return raw


def _get_n_frames(image):
    if isinstance(image, Image.Image):
        init_pos = image.tell()
        image.seek(0)
        n_frames = image.n_frames
        image.seek(init_pos)
    elif isinstance(image, tif.TiffFile):
        n_frames = len(image.pages)
    else:
        raise TypeError('Unsupported argument to get_n_frames; must be '
                        'PIL.Image or tifffile.TiffFile.')

    return n_frames

def _get_2d_shape(image):
    shape = None
    if isinstance(image, Image.Image):
        shape = image.size
    elif isinstance(image, tif.TiffFile):
        shape = image.series[0].shape[-2:]
    else:
        raise TypeError('Unsupported argument to get_2d_shape; must be '
                        'PIL.Image or tifffile.TiffFile.')

    return shape

def _get_frame_iter(image):
    frame_iter = None

    if isinstance(image, Image.Image):
        image.seek(0)
        frame_iter = ImageSequence.Iterator(image)
    elif isinstance(image, tif.TiffFile):
        frame_iter = iter(image.pages)
    else:
        raise TypeError('Unsupported argument to get_frame_iter; must be '
                        'PIL.Image or tifffile.TiffFile.')
    return frame_iter


def _get_frame_arr(frame, dtype=np.uint16):
    frame_arr = None

    if isinstance(frame, Image.Image):
        frame_arr = pil2numpy(frame, dtype=dtype)
    elif isinstance(frame, tif.TiffPage):
        frame_arr = frame.asarray()
    else:
        raise TypeError('Unsupported argument to get_frame_arr; must be '
                        'PIL.Image or tifffile.TiffPage.')
    return frame_arr


def parse_nd_image_hashes(image, format_class=ImHashV1):
    """
    Given a PIL.Image.Image or tifffile.TiffFile instance, retrieve its ImageDescription
    metadata and decode it according to format_class to produce a dict that can be used
    to assign 2D images to a position in an nd array. Also determines the implied shape of
    the image (except for the final 2 dimensions) from the metadata.
    """
    raw = _get_image_description(image)
    n_frames = _get_n_frames(image)

    out = decode_nd_hashes(raw, format_class=format_class)

    assert n_frames == len(out), \
        f'ImageDescription metadata had {len(out)} entries, but there are ' \
        f'{n_frames} frames in the image.'

    ndims = len(list(out.values())[0])
    shape = tuple([1 + max([v[dim] for v in out.values()])
                   for dim in range(ndims)
                   ])

    if 'shape' in raw:
        assert shape == tuple(raw['shape'][:-2])

    return out, shape


def hash_frames_to_ndarray(
        image,
        hashes,
        shape,
        format_class=ImHashV1,
        dtype=np.uint16
):
    assert isinstance(image, (tif.TiffFile, Image.Image))
    assert _get_n_frames(image) == len(hashes)

    full_shape = shape + _get_2d_shape(image)
    output = np.zeros(full_shape, dtype=dtype)

    for i, frame in enumerate(_get_frame_iter(image)):
        frame_arr = _get_frame_arr(frame)

        frame_hash = format_class.hash_frame(frame_arr)

        try:
            nd_index = hashes[frame_hash]
        except KeyError:
            raise ValueError(f'Frame {i} has hash value of {frame_hash},'
                             ' which is not present in the metadata dictionary that was saved.'
                             )
        output[nd_index] = frame_arr.copy()
        del frame_arr

    return output


def hash_imread(
        image,
        swapaxes=False,
        ensure_4d=False,
        backup=tif.imread,
        format_class=ImHashV1,
        dtype=np.uint16
):
    """
    Given a filename or TiffFile or PIL.Image, use the hashing system to read it
    in as a multidimensional TIFF image and return a properly arranged numpy ndarray.
    Will error if the image was not saved with properly encoded hash: indices information
    in its 'ImageDescription' tag (number 270), and if the same format_class is not used
    that was used to write the image.
    """

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, (tif.TiffFile, Image.Image)):
        pass
    else:
        raise TypeError('image must be string, Path, or PIL.Image.Image.')

    hashes, shape = parse_nd_image_hashes(image, format_class=format_class)

    imarr = hash_frames_to_ndarray(
        image,
        hashes,
        shape,
        format_class=format_class,
        dtype=dtype
    )

    if ensure_4d and imarr.ndim < 4:
        while imarr.ndim < 4:
            imarr = imarr[None]

    if swapaxes:
        imarr = imarr.swapaxes(0, 1)

    return imarr


def hash_imwrite(
        file,
        data,
        format_class=ImHashV1,
        **kwargs
):
    """
    Given a filename and an arbitrary numpy array with 2 or more dimensions, saves
    the array as a TIFF file with 2D frame hash values encoded in its ImageDescription
    tag via the `metadata` argument to tifffile.imwrite. The hash function and
    conversion between indices tuples and strings are defined by the format_class
    class methods hash_frame() and encode().
    """
    assert data.ndim >= 2, 'Supplied data must have at least 2 dimensions.'

    hashes = hash_nd_image(data, format_class=format_class)
    encoded_hashes = encode_nd_hashes(hashes, format_class=format_class)

    tif.imwrite(
        file,
        data,
        metadata=encoded_hashes,
        # This is essential to not assume 3-Z images are RGB
        photometric=tif.TIFF.PHOTOMETRIC.MINISBLACK,
        **kwargs
    )


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

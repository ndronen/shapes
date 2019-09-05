import sys
import os
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import cv2

import skimage
from skimage.draw import polygon as draw_polygon, circle as draw_circle
from skimage._shared.utils import warn

from skimage.draw._random_shapes import \
    _generate_rectangle_mask, _generate_circle_mask, \
    _generate_triangle_mask, _generate_random_colors, \
    SHAPE_GENERATORS, SHAPE_CHOICES


SHAPE_CLASSES = {'rectangle': 1, 'triangle': 2, 'circle': 3}


ObjectSpec = namedtuple(
    'ObjectSpec', ['generator', 'shape', 'fill', 'class_idx'])


def get_shape_name(shape_generator):
    if 'rectangle' in shape_generator.__name__:
        return 'rectangle'
    elif 'triangle' in shape_generator.__name__:
        return 'triangle'
    elif 'circle' in shape_generator.__name__:
        return 'circle'
    else:
        raise ValueError('Unknown shape generator {:s}'.format(
            shape_generator.__name__))


def generate_object_spec(shape_choices, *, textures=None, colors=None,
                         fill_is=None, class_is=None, random_state=None):
    """
    We also want to test a model's ability to segment shapes with
    respect to fill (color or texture) only, irrespective of shape. In
    this case, the class assigned to a shape should be determined by
    what it's filled by.
    """
    assert textures is None or colors is None
    assert not (textures is None and colors is None)
    assert fill_is in ['shape', 'random']
    assert class_is in ['shape', 'fill']
    assert random_state is not None

    shape_generator = random_state.choice(shape_choices)
    shape_name = get_shape_name(shape_generator)
    shape_class = SHAPE_CLASSES[shape_name]

    if textures is not None:
        assert len(textures) > 0
        fills = textures
    elif colors is not None:
        assert len(colors) > 0
        fills = colors

    if fill_is == 'shape':
        # The color of a shape is intrinsic to the shape, so the color
        # should be chosen from the provided colors using the class
        # index of the shape. (Decrement shape class by 1 to adjust
        # for the background class being class 0.)
        fill = fills[shape_class-1]

        # Because the mapping between shape and color is fixed, saying
        # that class of the object is the class of the shape is the
        # same as saying that the class of the object is the class of
        # the color.
        object_class = shape_class
    elif fill_is == 'random':
        # The color of a shape is accidental, so choose it randomly.
        fill_class = random_state.choice(len(fills))
        fill = fills[fill_class]

        # When the mapping between shape and color is random, we need
        # to know which attribute of the object determines the class.
        if class_is == 'shape':
            object_class = shape_class
        elif class_is == 'fill':
            # Add 1 to the color class because 0 is the background class.
            object_class = fill_class + 1

    spec = ObjectSpec(
        generator=shape_generator, shape=shape_name,
        fill=fill, class_idx=object_class)

    return spec


def remove_alpha_channel(image):
    if image.ndim == 3 and image.shape[-1] == 4:
        # Remove alpha channel
        image = image[:, :, :3]
    return image


def overlay_texture(image, mask, texture):
    if texture.ndim == 2:
        # Grayscale texture.
        t = texture[:, :, np.newaxis]
        image[mask, :] = t[mask]
    elif texture.ndim == 3:
        # RGB texture.
        image[mask] = texture[mask]
    else:
        raise ValueError(
            'Not sure how to handle {:d}-dim texture'.format(
                texture.ndim))

    return image


def overlay_color(image, mask, color):
    image[mask] = color
    return image


def overlay_object(image, target, filled, mask, object_spec):
    image = remove_alpha_channel(image)
    target = remove_alpha_channel(target)
    filled = remove_alpha_channel(filled)
    mask = remove_alpha_channel(mask)

    filled[mask] = True
    target[mask] = object_spec.class_idx

    # Conveniently resizing for now. Eventually, however, having control
    # of the scale at which textures appear in the image -- same scale per
    # object type, for instance -- would allow us to control the variance
    # in the generated datasets.
    fill = object_spec.fill

    assert isinstance(fill, np.ndarray)

    if fill.ndim in [2, 3]:
        return overlay_texture(image, mask, fill)
    elif fill.ndim == 1:
        return overlay_color(image, mask, fill)
    else:
        raise ValueError('Cannot handle {:d}-dimensional fill'.format(
            fill.ndim))


def random_shapes(image_shape, max_shapes, *, min_shapes=1, min_size=2,
                  max_size=None, multichannel=True, num_channels=3,
                  shape_names=None, intensity_range=None, allow_overlap=False,
                  num_trials=30, random_state=None, class_is=None,
                  fill_is=None, colors=None, textures=None,
                  background_texture=None):
    """
    Copied from scikit-image's `skimage.draw.random_shapes`.

    Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes
    bounded by `min_size` and `max_size`. It can occur that a randomly
    generated shape will not fit the image at all. In that case, the
    algorithm will try again with new starting coordinates a certain
    number of times. However, it also means that some shapes may be
    skipped altogether. In that case, this function will generate fewer
    shapes than requested.

    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    multichannel : bool, optional
        If True, the generated image has ``num_channels`` color channels,
        otherwise generates grayscale image.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate
        monochrome images, else color images with multiple
        channels. Ignored if ``multichannel`` is set to False.
    shape_names : {rectangle, circle, triangle, None} iterable of str, optional
        The name(s) of the shape(s) to generate or `None` to allow all shapes.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale
        images the format is (min, max). For multichannel - ((min, max),)
        if the ranges are equal across the channels, and ((min_0, max_0),
        ... (min_N, max_N)) if they differ. As the function supports
        generation of uint8 arrays only, the maximum range is (0,
        255). If None, set to (0, 254) for each channel reserving color
        of intensity = 255 for background.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before
        skipping it.
    seed : int, optional
        Seed to initialize the random number generator.  If `None`,
        a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.

    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    """

    assert class_is in ['shape', 'fill']
    assert fill_is in ['shape', 'random']

    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError(
            'Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    if not multichannel:
        num_channels = 1

    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    image_shape = (image_shape[0], image_shape[1], num_channels)
    if background_texture is None:
        image = np.full(image_shape, 255, dtype=np.uint8)
    else:
        image = background_texture.copy()
    target = np.full(image_shape, 0, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []
    masks = []

    num_shapes = random_state.randint(min_shapes, max_shapes + 1)

    # One random color per shape, unconnected to the shape itself. This
    # allows one to test a model's ability to segment the shapes with
    # respect to shape only, irrespective of color.
    if colors is None and textures is None:
        colors = _generate_random_colors(
            num_shapes, num_channels, intensity_range, random_state)

    # Create a list of (SHAPE, COLOR, CLASS) tuples.

    samples = []

    shape_choices = []
    if shape_names is None:
        shape_choices = SHAPE_CHOICES
    else:
        generator_map = {get_shape_name(sc): sc for sc in SHAPE_CHOICES}
        for shape_name in shape_names:
            shape_choices.append(generator_map[shape_name])

    for shape_num in range(num_shapes):
        object_spec = generate_object_spec(
            shape_choices,
            colors=colors, textures=textures, fill_is=fill_is,
            class_is=class_is, random_state=random_state)

        shape_size = (min_size, max_size)

        for trial_num in range(num_trials):
            # Pick start coordinates.
            column = random_state.randint(image_shape[1])
            row = random_state.randint(image_shape[0])
            point = (row, column)
            try:
                mask_idx, label = object_spec.generator(
                    point, image_shape, shape_size, random_state)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue

            print('Fit a shape')

            mask = np.zeros(image.shape[:2]).astype(np.uint8)
            cmask = mask.copy()
            mask[mask_idx] = 1

            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[mask].any():
                print('Going to overlay a shape')
                mask[mask_idx] = 255
                print('findContours')
                cv2.imwrite('mask.png', mask)
                print('mask', mask.min(), mask.max(), file=sys.stderr)
                _, contours, _ = cv2.findContours(
                    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                print('drawContours', file=sys.stderr)
                cmask = cv2.drawContours(cmask, contours, -1, (255,255,255), 1)
                cmask = mask.astype(bool)
                # Calling `overlay_object` has side effects, such
                # as setting the pixels of `filled` to `True` where
                # the object exists.
                image = overlay_object(
                    image, target, filled, cmask, object_spec)
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if not multichannel:
        image = np.squeeze(image, axis=2)

    return image, labels, target

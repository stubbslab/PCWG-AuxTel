# This file forms the backbone for the alternative analysis of "donuts"
# from AuxTel. It is based on work initially done by Chris Stubbs.
import cv2
import lsst.rapid.analysis.butlerUtils as butlerUtils
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from lsst.rapid.analysis.imageExaminer import ImageExaminer
import os


def findCircles(day_obs, seq_nums, doPlot=False, planeSkew=False, config=None, path=None,
                butler_location='NCSA', **kwargs):
    """This finds all the circles! this function simply loops over a
    sequence list (seq_nums) the function FindCircle, which does all
    the work for a single exposure.

    Parameters
    ----------
    day_obs : `string`
        The observation date we are looking to find circles for.

    seq_nums : `iterable`
        The iterable of sequences we are hoping to obtain exposures for.

    doPlot : `bool`
        Whether or not we should attempt to plot all supporting figures
        along the way.

    config : `dict`
        Optional configuration dictionary, if not provided we will use a
        standard configuration defined below.

    path : `string`
        Optional path where detail plots should be saved.

    butler_location : `string`
        Optional input, to change default butler type

    Returns
    -------
    efd_infos : `list of dicts`
        List of dicts of all relevant EFD data related to the exposures.

    dxs : `list`
        The displacements along the x-axis for all exposures.

    dys : `list`
        The displacements along the y-axis for all exposures.

    coefficients : `list`
        Optional list of plane coefficients for flux skew.
    """
    butler = butlerUtils.makeDefaultLatissButler(butler_location)
    efd_infos = []
    dxs = []
    dys = []
    coefficients = []

    # Here we look if a config dictionary is passed, if not then we will
    # use the default values.
    if not config:
        config = {"halfbox": 1200, "kernel": 61, "minclip": 0.1, "maxclip": 0.5,
                  "outerRadius": 750, "innerRadius": 300, "vmin": 10,
                  "vmax": 500, "normPercent": 85, "skyPercent": 10,
                  "minDistOuter": 200, "minDistInner": 300
                  }

    if not path:
        path = os.path.join(os.path.expanduser('~'), f"detail_plots_{day_obs}")

    if doPlot:
        doPlot = _pathcheck(path)

    for seq_num in seq_nums:
        outer_circle = np.zeros(3, dtype=int)
        inner_circle = np.zeros(3, dtype=int)
        dataId = {'day_obs': day_obs, 'seq_num': seq_num, 'detector': 0}
        exp = butler.get('quickLookExp', dataId)
        outer_circle, inner_circle, coef = findCircle(exp, config, seq_num, path, doPlot, planeSkew)
        if planeSkew:
            coefficients.append(coef)

        centration_offset = outer_circle - inner_circle
        print(f"Seq_num: {seq_num}, dx_offset={centration_offset[0,0]}, dy_offset={centration_offset[0,1]}")
        expId = {}
        expId, position = get_efd_info(day_obs, seq_num, butler)
        expId.update(position)
        efd_infos.append(expId)
        dxs.append(centration_offset[0, 0])
        dys.append(centration_offset[0, 1])

    return efd_infos, dxs, dys, coefficients


def findCircle(exp, config, seqNum, path, doPlot=False, doPlaneSkew=False, useCutout=False):
    """This function does all the tricks to find the circle for a single
    exposure and returns the inner and outer circles for it.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure we are attempting to find the circles for.

    config : `dict`
        Dictionary of the configuration options needed throughout the code.

    seqNum : `integer`
        The sequence number, this is purely needed for the plotting part, but
        will be asked for everytime.

    path : `string`
        Path where the extra plots will be saved. Ignored if doPlot=False.

    doPlot : `bool`
        Whether we should do all the extra plotting or not.

    doPlaneSkew : `bool`
        Whether or not we should attempt to fit the flux to a plane.

    useCutout : `bool`
        Wheter to try and use the cutout feature to reduce image size.

    Returns
    -------
    outer_circle : `list`
        Consisting of the cetroid position x, y and the radius
        of the outer circle of the donut.

    inner_circle : `list`
        Consisting of the centroid position x, y and the radius
        of the inner circle of the donut.

    plane_coefficient : `list`, optional
        list of the coefficients (C) needed to plot a plane.
        z = C[0]*x = C[1]*y + C[2]
        Will be np.Nan's if doPlaneSkew=False.
    """
    path = os.path.join(path, f"seq{seqNum:05}")
    if doPlot:
        doPlot = _pathcheck(path)

    if useCutout:
        imexam = _examine(exp, config, path, doPlot)
        image = _cutOut(imexam, config, path, doPlot)
    else:
        image = np.array(exp.image.array)

    norm_image, cutout_smoothed = _getSmoothedAndNormalizedImages(image, config['kernel'],
                                                                  config['normPercent'], config['skyPercent'],
                                                                  path, doPlot)

    # Now we have the normalized image, and we want to convert to a mask of
    # which part of the image we want to fit to.

    int_image = _makeIntegerMask(norm_image, config['maxclip'], path, doPlot)

    # Now calling cv2 to find the actual circles using a HoughCircles
    # transform.

    # Here, param1 and param2 are related to to how the cv2 method finds the
    # edges of the circle, and the center respectively.
    params_big = {'param1': 10, 'param2': 10, 'minRadius': int(config['outerRadius']),
                  'maxRadius': int(1.2*config['outerRadius'])}
    params_small = {'param1': 30, 'param2': 10, 'minRadius': int(config['innerRadius']),
                    'maxRadius': int(1.2*config['innerRadius'])}

    outer_circle = np.empty(3)
    inner_circle = np.empty(3)
    outer_circle = _applyHoughTransform(int_image, config['minDistOuter'], params_big)
    inner_circle = _applyHoughTransform(int_image, config['minDistInner'], params_small)
    path2 = path
    if doPlot:
        path = os.path.join(path, "circlefits.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        for (x, y, r) in outer_circle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(int_image, (x, y), r, (128, 128, 128), 20)
            cv2.rectangle(int_image, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 128), -10)
        ax1.imshow(int_image, origin='lower')
        for (x, y, r) in inner_circle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(int_image, (x, y), r, (128, 128, 128), 20)
            cv2.rectangle(int_image, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 0), -10)
        ax2.imshow(int_image, origin='lower')
        fig.savefig(path)

    if doPlaneSkew:
        plane_coefficient = _findPlaneSkew(cutout_smoothed, norm_image, config['maxclip'], path2, doPlot)
    else:
        plane_coefficient = np.NaN

    return outer_circle, inner_circle, plane_coefficient


def _examine(exp, config, path, doPlot=False):
    '''This function has been deprecated. It is only called if we need to
    cutout the donut from a larger set of data.'''
    path = os.path.join(path, "detail1.png")
    imexam = ImageExaminer(exp, boxHalfSize=config["halfbox"], savePlots=path)
    if doPlot:
        imexam.plot()
    return imexam


def _cutOut(imexam, config, path, doPlot=False):
    '''This function has been deprecated. It is only called if we need to
    cutout the donut from a larger set of data.'''
    cutout = np.array(imexam.data)
    if doPlot:
        path = os.path.join(path, "detail2.png")
        fig = plt.figure()
        plt.imshow(cutout, cmap='gray', origin='lower', vmin=config['vmin'], vmax=config['vmax'])
        fig.savefig(path)
    return cutout


def _getSmoothedAndNormalizedImages(cutout, kernel, normPercent, skyPercent, path, doPlot=False):
    # We smooth the cutout
    cutoutSmoothed = cv2.GaussianBlur(cutout, (kernel, kernel), 0)

    halfbox = int(cutout.shape[0]/2)
    if doPlot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        ax1.imshow(cutoutSmoothed, origin='lower')
        ax2.plot(cutoutSmoothed[halfbox, :])

    # We normalize and remove background sky values:
    normValue = np.percentile(cutoutSmoothed, normPercent)
    skyValue = np.percentile(cutoutSmoothed, skyPercent)

    normImage = (cutoutSmoothed-skyValue)/normValue

    if doPlot:
        path = os.path.join(path, "detail3.png")
        ax3.imshow(normImage, origin='lower')
        ax4.plot(normImage[halfbox, :])
        fig.savefig(path)

    return normImage, cutoutSmoothed


def _makeIntegerMask(normImage, maxclip, path, doPlot):
    # Adding the option that we can either use a predefined value,
    # or we can use an automatically calculated clip.
    if not maxclip:
        max_img = np.max(normImage)
        mean_img = np.mean(normImage)
        maxclip = max_img/2 - mean_img

    normMask = ma.getmask(ma.masked_greater_equal(normImage, maxclip))
    intImage = np.uint8(255*normMask)

    if doPlot:
        path = os.path.join(path, "integerMask.png")
        fig = plt.figure()
        plt.imshow(intImage, origin='lower', cmap='RdGy')
        plt.colorbar()
        fig.savefig(path)

    return intImage


def _applyHoughTransform(intImage, min_dist, params):
    circle = cv2.HoughCircles(intImage, cv2.HOUGH_GRADIENT, 1, min_dist, **params)

    circle = np.round(circle[0, :]).astype(int)
    return circle


def _findPlaneSkew(smoothedImage, normImage, maxclip, path, doPlot):
    """What if we wanted to also find the skew, of the flux plane? Here we
    attempt to fit a plane to the normalized fluxes, so in case we do see that
    the flux is uneven across the image, we have the coefficients for a
    z = a*x + b*y + c plane. with z being the flux.
    This implementation is heavily inspired by the example given in
    https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    """
    if not maxclip:
        max_img = np.max(normImage)
        mean_img = np.mean(normImage)
        maxclip = max_img/2 - mean_img
    nmi = ma.masked_less_equal(normImage, maxclip)

    if doPlot:
        path = os.path.join(path, "cutout.png")
        fig = plt.figure()
        plt.imshow(nmi, origin='lower')
        plt.colorbar()
        fig.savefig(path)

    grid = np.indices(nmi.shape)
    mask = ma.getmask(nmi)
    y = ma.array(grid[0], mask=mask)
    x = ma.array(grid[1], mask=mask)
    z = ma.array(normImage, mask=mask)
    data = np.c_[x.compressed(), y.compressed(), z.compressed()]
    a = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    import scipy.linalg as linalg
    c, _, _, _ = linalg.lstsq(a, data[:, 2])
    return c


def _pathcheck(path):
    try:
        os.makedirs(path, exist_ok=True)
        doPlot = True
    except FileExistsError:
        doPlot = False
        print(f"Seems like a file exists at {path}, so we can't make a folder.")
        print(" We have therefore turned off plotting.")
    except PermissionError:
        doPlot = False
        print(f"We cannot save the files to {path}, we lack permission.")
        print(" We have therefore turned off plotting.")
    return doPlot


def _getEfdData(client, dataSeries, startTime, endTime):
    """A synchronous wrapper for geting the data from the EFD.
    This exists so that the top level functions don't all have to be async def.
    curtesy of Merlin Levine-Fisher.
    """
    import asyncio
    import nest_asyncio
    # This is the magic that let's us call this asyncio loop from inside a
    # jupyternotebook
    nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(client.select_time_series(dataSeries, ['*'], startTime, endTime))


def get_efd_info(dayObs, seqNum, butler):
    """ Wrapper that grabs the EFD info for each sequence"""
    from lsst_efd_client import EfdClient
    from lsst.rapid.analysis.butlerUtils import getExpIdFromDayObsSeqNum

    client = EfdClient('summit_efd')

    dataId = {'day_obs': dayObs, 'seq_num': seqNum}
    expId = getExpIdFromDayObsSeqNum(butler, dataId)
    where = "exposure.day_obs=day_obs AND exposure.seq_num=seq_num"
    expRecords = butler.registry.queryDimensionRecords("exposure", where=where,
                                                       bind={'day_obs': dataId['day_obs'],
                                                             'seq_num': dataId['seq_num']})
    expRecords = list(expRecords)
    assert len(expRecords) == 1, f'Found more than one exposure record for {dataId}'
    record = expRecords[0]
    t_start = record.timespan.begin
    t_end = record.timespan.end

    hex_position = _getEfdData(client, "lsst.sal.ATHexapod.positionStatus", t_start.utc, t_end.utc)
    # This dictionary gives the hexapod names and indices
    # units for x,y,z in mm and u,v,w in degrees, according to
    # https://ts-xml.lsst.io/sal_interfaces/ATHexapod.html#positionupdate.

    names = ['x', 'y', 'z', 'u', 'v', 'w']
    positions = {}
    for val, name in enumerate(names):
        key = f'reportedPosition{val}'
        position = hex_position[key][0]
        positions[name] = position

    return expId, positions

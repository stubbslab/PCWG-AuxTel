# This file forms the backbone for the alternative analysis of "donuts"
# from AuxTel. It is based on work initially done by Chris Stubbs.
import cv2
import lsst.daf.persistence as dafPersist
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from lsst.rapid.analysis.imageExaminer import ImageExaminer
import os


def findCircles(day_obs, seq_nums, doPlot=False, config=None, path=None, **kwargs):
    """Let's find all the circles! this function simply loops over a
    sequence list (seq_List) the function FindCircle, which does all
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

    Returns
    -------
    efd_infos : `list of touples`
        List of all relevant EFD data related to the exposures.

    dxs : `list`
        The displacement along the x-axis for all exposures.

    dys : `list`
        The dispalcement along the y-axis for all exposures.
    """
    butler = dafPersist.Butler('/project/shared/auxTel/rerun/quickLook')
    # efd_infos = []
    dxs = []
    dys = []

    # Here we look if a config dictionary is passed, if not then we will
    # use the default values.
    if not config:
        config = {"Halfbox": 1200, "kernel": 61, "minclip": 0.1, "maxclip": 0.5,
                  "outer_radius": 750, "inner_radius": 300, "vmin": 10,
                  "vmax": 500, "normPercent": 85, "skyPercent": 10,
                  "min_dist_outer": 200, "min_dist_inner": 300
                  }

    if not path:
            path = os.path.join(os.getcwd(), "detail_plots")

    if doPlot:
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            print(f"Seems like a file exists at {path}, so we can't make a folder.")
        except PermissionError:
            print(f"We cannot save the files to {path}, we lack permission.")

    for seq_num in seq_nums:
        outer_circle = np.zeros(3, dtype=int)
        inner_circle = np.zeros(3, dtype=int)
        exp = butler.get('quickLookExp', dayObs=day_obs, seqNum=seq_num)
        outer_circle, inner_circle = findCircle(exp, config, seq_num, path, doPlot)
        centration_offset = outer_circle - inner_circle
        print(f"Seq_num: {seq_num}, dx_offset={centration_offset[0,0]}, dy_offset={centration_offset[0,1]}")
        # expId, position = get_efd_info(obs_Date, seq_Num)

        # efd_infos.append([expId, position])
        dxs.append(centration_offset[0, 0])
        dys.append(centration_offset[0, 1])

    return dxs, dys


def findCircle(exp, config, seqNum, path, doPlot=False):
    """This function finds does all the tricks to find the circle for a single exposure
    and returns the inner and outer circles for it.

    Parameters
    ----------
    exp :
        The exposure we are attempting to find the circles for.

    config : 'dict'
        Dictionary of the configuration options needed throughout the code.

    seqNum : 'integer'
        The sequence number, this is purely needed for the plotting part, but will
        be asked for everytime.

    path : 'string'
        string, showing path where the extra plots would be saved.

    doPlot : 'bool'
        Boolean, whether we should do all the extra plotting or not.

    Returns
    -------
    outer_circle : 'list'
        list consisting of the cetroid position (x,y) and the radius of the outer
        circle of the donut.

    inner_circle : 'list'
        list consisting of the centroid position (x,y) and the radius of the inner
        circle of the donut.
    """
    path = os.path.join(path, f"seq{seqNum}")
    if doPlot:
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            print(f"Seems like a file exists at {path}, so we can't make a folder.")
        except PermissionError:
            print(f"We cannot save the files to {path}, we lack permission.")

    imexam = _examine(exp, config, path, doPlot)

    cutout = _cutOut(imexam, config, path, doPlot)

    print("We cut out the figure, next lets smooth it", flush=True)

    norm_image = _smoothNormalized(cutout, config, path, doPlot)

    # Now we have the normalized image, and we want to convert those to either being there or not

    int_image = _detectMask(norm_image, config, path, doPlot)

    # Now calling cv2 to find the actual circles using a HoughCircles.

    params_big = {'param1': 10, 'param2': 10, 'minRadius': int(config['outer_radius']),
                  'maxRadius': int(1.2*config['outer_radius'])}
    params_small = {'param1': 30, 'param2': 10, 'minRadius': int(config['inner_radius']),
                    'maxRadius': int(1.2*config['inner_radius'])}

    outer_circle = np.empty(3, dtype=int)
    inner_circle = np.empty(3, dtype=int)
    outer_circle = _applyHoughTransform(int_image, config['min_dist_outer'], params_big)
    inner_circle = _applyHoughTransform(int_image, config['min_dist_inner'], params_small)

    if doPlot:
        path = os.path.join(path, "detail5.png")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for (x, y, r) in outer_circle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(int_image, (x, y), r, (128, 128, 128), 4)
            cv2.rectangle(int_image, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 128), -1)
        ax1.imshow(int_image, origin='lower')
        for (x, y, r) in inner_circle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(int_image, (x, y), r, (128, 128, 128), 4)
            cv2.rectangle(int_image, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 0), -1)
        ax2.imshow(int_image, origin='lower')
        fig.savefig(path)

    return outer_circle, inner_circle


def _examine(exp, config, path, doPlot=False):
    path = os.path.join(path, "detail1.png")
    imexam = ImageExaminer(exp, boxHalfSize=config["Halfbox"], savePlots=path)
    if doPlot:
        imexam.plot()
    return imexam


def _cutOut(imexam, config, path, doPlot=False):
    cutout = np.array(imexam.data)
    if doPlot:
        path = os.path.join(path, "detail2.png")
        fig = plt.figure()
        plt.imshow(cutout, cmap='gray', origin='lower', vmin=config['vmin'], vmax=config['vmax'])
        fig.savefig(path)
    return cutout


def _smoothNormalized(cutout, config, path, doPlot=False):
    # We smooth the cutout
    cutoutSmoothed = cv2.GaussianBlur(cutout, (config["kernel"], config["kernel"]), 0)

    if doPlot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(cutoutSmoothed, origin='lower')
        ax2.plot(cutoutSmoothed[config['Halfbox'], :])

    # We normalize and remove background sky values:
    normValue = np.percentile(cutoutSmoothed, config["normPercent"])
    skyValue = np.percentile(cutoutSmoothed, config["skyPercent"])

    normImage = (cutoutSmoothed-skyValue)/normValue

    if doPlot:
        path = os.path.join(path, "detail3.png")
        ax3.plot(normImage[config['Halfbox'], :])
        fig.savefig(path)

    return normImage


def _detectMask(normImage, config, path, doPlot):
    normMask = ma.getmask(ma.masked_greater_equal(normImage, config["maxclip"]))
    intImage = np.uint8(255*normMask)

    if doPlot:
        path = os.path.join(path, "detail4.png")
        fig = plt.figure()
        plt.imshow(intImage, origin='lower', cmap='RdGy')
        plt.colorbar()
        fig.savefig(path)

    return intImage


def _applyHoughTransform(intImage, min_dist, params):
    circle = cv2.HoughCircles(intImage, cv2.HOUGH_GRADIENT, 1, min_dist, **params)

    circle = np.round(circle[0, :]).astype(int)
    return circle


def _getEfdData(client, dataSeries, startTime, endTime):
    import asyncio
    """A synchronous warpper for geting the data from the EFD.
    This exists so that the top level functions don't all have to be async def.
    curtesy of Merlin Levine-Fisher.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(client.select_time_series(dataSeries, ['*'], startTime, endTime))


def get_efd_info(dayObs, seqNum, butler):
    """ Wrapper that grabs the EFD info for each sequence, currently this
    wrapper does not work, there seems to be an issue with asyncio."""
    from astropy.time import Time, TimeDelta
    from lsst_efd_client import EfdClient

    client = EfdClient('summit_efd')

    dataId = {'dayObs': dayObs, 'seqNum': seqNum}
    expId = butler.queryMetadata('raw', 'expId', **dataId)[0]

    tStart = butler.queryMetadata('raw', ['DATE'], detector=0, expId=expId)  # Get the data
    t_start = Time(tStart, scale='tai')[0]
    t_end = t_start + TimeDelta(1, format='sec')

    # Get the reported position
    # hex_position = client.select_time_series("lsst.sal.ATHexapod.positionStatus", ['*'],t_start, t_end)
    # hex_position = _getEfdData(client, "lsst.sal.ATHexapod.positionStatus", t_start, t_end)
    # Wild attempt from my side:
    hex_position = client.select_time_series("lsst.sal.ATHexapod.positionStatus", ['*'], t_start, t_end)
    print(hex_position.head())
    # This dictionary gives the hexapod names and indices
    # units for x,y,z in mm and u,v,w in degrees, according to
    # https://ts-xml.lsst.io/sal_interfaces/ATHexapod.html#positionupdate.

    names = {'u': 3, 'v': 4, 'w': 5, 'x': 0, 'y': 1, 'z': 2}
    positions = {}
    for name in names.keys():
        key = 'reportedPosition%d'%names[name]
        position = hex_position[key][0]
        positions[name] = position

    return expId, positions

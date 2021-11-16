# This file forms the backbone for the alternative analysis of "donuts"
# from AuxTel. It is based on work initially done by Chris Stubbs.
import cv2
import lsst.daf.persistence as dafPersist
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from lsst.rapid.analysis.imageExaminer import ImageExaminer
import os
import logging 

def findCircles(day_obs, seq_nums, doPlot=False, planeSkew=False, config=None, path=None, **kwargs):
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

    path : `string`
        Optional path where detail plots should be saved. 

    Returns
    -------
    efd_infos : `list of touples`
        List of all relevant EFD data related to the exposures.

    dxs : `list`
        The displacement along the x-axis for all exposures.

    dys : `list`
        The displacement along the y-axis for all exposures.
    
    coefficients : `list`
        Optional list of coefficients for flux skew. 
    """
    butler = dafPersist.Butler('/project/shared/auxTel/rerun/quickLook')
    efd_infos = []
    dxs = []
    dys = []
    if planeSkew:
        coef = []

    # Here we look if a config dictionary is passed, if not then we will
    # use the default values.
    if not config:
        config = {"Halfbox": 1200, "kernel": 61, "minclip": 0.1, "maxclip": 0.5,
                  "outer_radius": 750, "inner_radius": 300, "vmin": 10,
                  "vmax": 500, "normPercent": 85, "skyPercent": 10,
                  "min_dist_outer": 200, "min_dist_inner": 300
                  }

    if not path:
            path = os.path.join(os.getcwd(), f"detail_plots_{day_obs}")

    if doPlot:
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            doPlot = False
            print(f"Seems like a file exists at {path}, so we can't make a folder.")
            print(" We have therefore turned of plotting.")
        except PermissionError:
            doPlot = False
            print(f"We cannot save the files to {path}, we lack permission.")
            print(" We have therefore turned of plotting.")

    for seq_num in seq_nums:
        outer_circle = np.zeros(3, dtype=int)
        inner_circle = np.zeros(3, dtype=int)
        exp = butler.get('quickLookExp', dayObs=day_obs, seqNum=seq_num)
        if planeSkew:
            outer_circle, inner_circle, coefficients = findCircle(exp, config, seq_num, path, doPlot, planeSkew)
            coef.append(coefficients)
        else:    
            outer_circle, inner_circle = findCircle(exp, config, seq_num, path, doPlot)
        centration_offset = outer_circle - inner_circle
        print(f"Seq_num: {seq_num}, dx_offset={centration_offset[0,0]}, dy_offset={centration_offset[0,1]}")
        expId, position = get_efd_info(obs_Date, seq_Num)

        efd_infos.append([expId, position])
        dxs.append(centration_offset[0, 0])
        dys.append(centration_offset[0, 1])
    if planeSkew:
        return dxs, dys, coef, efd_infos 
    else:
        return dxs, dys, efd_infos


def findCircle(exp, config, seqNum, path, doPlot=False, planeSkew=False, useCutout=False):
    """This function does all the tricks to find the circle for a single exposure
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

    planeSkew : 'bool'
        Boolean, wheter or not we should attempt to fit the flux to a plane.
    
    useCutout ; 'bool'
        Boolean, wheter to try and use the cutout feature to reduce image size.

    Returns
    -------
    outer_circle : 'list'
        list consisting of the cetroid position (x,y) and the radius of the outer
        circle of the donut.

    inner_circle : 'list'
        list consisting of the centroid position (x,y) and the radius of the inner
        circle of the donut.

    (Optional) plane_coefficient : 'list'
        list of the coefficients (C) needed to plot a plane z = C[0]*x = C[1]*y + C[2]
    """
    path = os.path.join(path, f"seq{seqNum:05}")
    if doPlot:
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            doPlot = False
            print(f"Seems like a file exists at {path}, so we can't make a folder.")
        except PermissionError:
            doPlot = False
            print(f"We cannot save the files to {path}, we lack permission.")

    if useCutout:
        imexam = _examine(exp, config, path, doPlot)
        image = _cutOut(imexam, config, path, doPlot)
    else:
        image = np.array(exp.image.array)

    norm_image, cutout_smoothed = _smoothNormalized(image, config, path, doPlot)

    # Now we have the normalized image, and we want to convert those to either being there or not

    int_image = _detectMask(norm_image, config, path, doPlot)

    # Now calling cv2 to find the actual circles using a HoughCircles transform.

    # Here, param1 and param2 are related to to how the cv2 method finds the edges of the circle, 
    # and the center respectively. 
    params_big = {'param1': 10, 'param2': 10, 'minRadius': int(config['outer_radius']),
                  'maxRadius': int(1.2*config['outer_radius'])}
    params_small = {'param1': 30, 'param2': 10, 'minRadius': int(config['inner_radius']),
                    'maxRadius': int(1.2*config['inner_radius'])}

    outer_circle = np.empty(3)
    inner_circle = np.empty(3)
    outer_circle = _applyHoughTransform(int_image, config['min_dist_outer'], params_big)
    inner_circle = _applyHoughTransform(int_image, config['min_dist_inner'], params_small)
    path2 = path
    if doPlot:
        path = os.path.join(path, "detail5.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,10))
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

    if planeSkew:
        plane_coefficient = _planeskew(cutout_smoothed, norm_image, config, path2, doPlot)
        return outer_circle, inner_circle, plane_coefficient
    else:
        return outer_circle, inner_circle


def _examine(exp, config, path, doPlot=False):
    '''This function has been deprecated and should be deleted.'''
    path = os.path.join(path, "detail1.png")
    imexam = ImageExaminer(exp, boxHalfSize=config["Halfbox"], savePlots=path)
    if doPlot:
        imexam.plot()
    return imexam


def _cutOut(imexam, config, path, doPlot=False):
    '''This function has been deprecated and should be deleted.'''
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

    halfbox = int(cutout.shape[0]/2)
    if doPlot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10,10))
        ax1.imshow(cutoutSmoothed, origin='lower')
        ax2.plot(cutoutSmoothed[halfbox, :])

    # We normalize and remove background sky values:
    normValue = np.percentile(cutoutSmoothed, config["normPercent"])
    skyValue = np.percentile(cutoutSmoothed, config["skyPercent"])

    normImage = (cutoutSmoothed-skyValue)/normValue

    if doPlot:
        path = os.path.join(path, "detail3.png")
        ax3.imshow(normImage, origin='lower')
        ax4.plot(normImage[halfbox, :])
        fig.savefig(path)

    return normImage, cutoutSmoothed


def _detectMask(normImage, config, path, doPlot):
    # Adding the option that we can either use a predefined value,
    # or we can use an automatically calculated clip.
    if "maxclip" in config.keys():
        clip = config["maxclip"]
    else:
        max_img = np.max(normImage)
        mean_img = np.mean(normImage)
        clip = max_img/2 - mean_img
    normMask = ma.getmask(ma.masked_greater_equal(normImage, clip))
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

def _planeskew(smoothedImage, normImage, config, path, doPlot):
    """ What if we wanted to also find the skew, of the flux plane? Here we attempt to fit
    a plane to the normalized fluxes, so in case we do see that the flux is uneven accross
    the image, we have the coefficients for a z = a*x + b*y + c plane. with z being the flux.
    This implementation is heavily inspired by the example given in 
    https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    """
    if "maxclip" in config.keys():
        clip = config["maxclip"]
    else:
        max_img = np.max(normImage)
        mean_img = np.mean(normImage)
        clip = max_img/2 - mean_img
    nmi = ma.masked_less_equal(normImage, clip)

    if doPlot:
        path = os.path.join(path, "cutout.png")
        fig = plt.figure()
        plt.imshow(nmi, origin = 'lower')
        plt.colorbar()
        fig.savefig(path)

    grid = np.indices(nmi.shape)
    mask = ma.getmask(nmi)
    y = ma.array(grid[0], mask = mask)
    x = ma.array(grid[1], mask = mask)
    z = ma.array(normImage, mask = mask)
    data = np.c_[x.compressed(),y.compressed(),z.compressed()]
    A = np.c_[data[:,0],data[:,1],np.ones(data.shape[0])]
    import scipy.linalg as linalg
    C,_,_,_ = linalg.lstsq(A,data[:,2])
    return C


def _getEfdData(client, dataSeries, startTime, endTime):
    import asyncio
    """A synchronous wrapper for geting the data from the EFD.
    This exists so that the top level functions don't all have to be async def.
    curtesy of Merlin Levine-Fisher.
    """
    loop = asyncio.get_event_loop()
    future = asyncio.run_coroutine_threadsafe(client.select_time_series(dataSeries, ['*'], startTime, endTime), loop)
    return future.result(timeout=10)


def get_efd_info(dayObs, seqNum, butler):
    """ Wrapper that grabs the EFD info for each sequence, currently this
    wrapper does not work, there seems to be an issue with asyncio."""
    from astropy.time import Time, TimeDelta
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
    
    #hex_position = client.select_time_series("lsst.sal.ATHexapod.positionStatus", ['*'], t_start, t_end)
    hex_position = _getEfdData(client, "lsst.sal.ATHexapod.positionStatus",t_start, t_end)
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

# This file forms the backbone for the alternative analysis of "donuts"
# from AuxTel. It is based on work initially done by Chris Stubbs.
import cv2
import lsst.daf.persistence as dafPersist
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
# import copy
from lsst.rapid.analysis.imageExaminer import ImageExaminer
import os


def Findcircles(obs_Date, seq_List, do_plot=0, config="None"):
    """Let's find all the circles! this function simply loops over a
    sequence list (seq_List) the function FindCircle, which does all
    the work for a single exposure.

    Parameters
    ----------
    obs_Date : `string`
        The observation date we are looking to find circles for.

    seq_List : `list`
        The list of sequences we are hoping to obtain exposures for.

    do_plot : `bool`
        Whether or not we should be attempt to plot all supporting figures
        along the way.

    config : `dict`
        Optional configuration dictionary, if not provided we will use a
        standard configuration defined below.

    Returns
    -------
    EFDlist : `list`
        List of all relevant EFD data related to the exposures.

    dxArray : `list`
        The displacement along the x-axis for all exposures.

    dyArray : `list`
        the dispalcement along the y-axis for all exposures.
    """
    butler = dafPersist.Butler('/project/shared/auxTel/rerun/quickLook')
    # EFDlist = []
    dxArray = []
    dyArray = []

    ''' Here we look if a config dictionary is passed, if not then we will
    use the default values. '''
    if config != "None":
        config = config
    else:
        config = {"Halfbox": 1200, "kernel": 61, "minclip": 0.1,
                  "maxclip": 0.5, "outer_radius": 750, "inner_radius": 300}

    if do_plot:
        if os.path.isdir('detail_plots'):
            pass
        else:
            os.mkdir('detail_plots')

    for seq_Num in seq_List:
        outercircle = np.zeros(3, dtype=int)
        innercircle = np.zeros(3, dtype=int)
        exp = butler.get('quickLookExp', dayObs=obs_Date, seqNum=seq_Num)
        outercircle, innercircle = FindCircle(exp, config, seq_Num, do_plot)
        centrationoffset = outercircle - innercircle
        print(f"Seq_num: {seq_Num}, dx_offset={centrationoffset[0,0]}, dy_offset={centrationoffset[0,1]}")
        # expId, position = getEFDinfo(obs_Date, seq_Num)

        # EFDlist.append([expId, position])
        dxArray.append(centrationoffset[0, 0])
        dyArray.append(centrationoffset[0, 1])

    return dxArray, dyArray


def FindCircle(exp, config, seqNum, do_plot=0):
    """ This function finds does all the tricks to find the circle, and returns
    the inner and outer circles
    """
    if do_plot:
        path = f'detail_plots/seq{seqNum}'
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)

    def examine(exp):
        imexam = ImageExaminer(exp, boxHalfSize=config["Halfbox"])
        if do_plot == 1:
            fig = plt.figure()
            imexam.plot()
            fig.savefig(path+'/detail1.png')
        return imexam

    def cut_out(imexam):
        cutout = np.array(imexam.data)
        if do_plot == 1:
            fig = plt.figure()
            plt.imshow(cutout, cmap='gray', origin='lower', vmin=10, vmax=500)
            fig.savefig(path+'/detail2.png')
        return cutout

    ''' Not sure how much we need to make individual functions, seeing as the
    work flow is the same for each image in the sequence.
    '''

    imexam = examine(exp)

    cutout = cut_out(imexam)

    print("We cut out the figure, next lets smooth it", flush=True)
    # We smooth the cutout
    cutoutSmoothed = cv2.GaussianBlur(cutout, (config["kernel"], config["kernel"]), 0)

    if do_plot == 1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(cutoutSmoothed, origin='lower')
        ax2.plot(cutoutSmoothed[1000, :])

    # We normalize and remove background sky values:
    normvalue = np.percentile(cutoutSmoothed, 85)
    skyvalue = np.percentile(cutoutSmoothed, 10)

    normimage = (cutoutSmoothed-skyvalue)/normvalue

    if do_plot == 1:
        ax3.plot(normimage[1000, :])
        fig.savefig(path+'/detail3.png')

    # Now we have the normalized image, and we want to convert those to either being there or not

    # Think this might be an easier way of getting the 0/1 data out.
    normmask = ma.getmask(ma.masked_greater_equal(normimage, config["maxclip"]))
    intimage = np.uint8(255*normmask)

    """ This is what Chris's code looked like for this part, I am a little confused about what he does,
    there is in particular the issue of what happens when he computes the intimage from the normimage,
    these two are close, but something seems to happen at the edges? I did check, the intimage I get
    with the code above is equal to the intimage that Chris method gets as well, via. a np.array_equal"""
    # np.clip(normimage, configs["minclip"], configs["maxclip"] , out=normimage)
    # set regions above maxclip to maxclip and below minclip to minclip
    # normimage[normimage<=configs["minclip"]]=0
    # normimage[normimage>=configs["maxclip"]]=255

    # intimage=np.uint8(normimage*255/(max(np.ndarray.flatten(normimage))))

    # copying it
    # origintimage = copy.deepcopy(intimage)  # keep a pristine one for later

    if do_plot == 1:
        fig = plt.figure()
        plt.imshow(intimage, extent=[0, 2*config["Halfbox"], 0, 2*config["Halfbox"]], origin='lower',
                   cmap='RdGy')
        plt.colorbar()
        fig.savefig(path+'/detail4.png')
        # plt.plot(intimage[1200,:])

    # Now calling cv2 for things first the outer circle!

    def get_circle(intimage, mindist, params):
        circle = cv2.HoughCircles(intimage, cv2.HOUGH_GRADIENT, 1, mindist, **params)

        circle = np.round(circle[0, :]).astype(int)
        return circle

    params_big = {'param1': 10, 'param2': 10, 'minRadius': int(config['outer_radius']),
                  'maxRadius': int(1.2*config['outer_radius'])}
    params_small = {'param1': 30, 'param2': 10, 'minRadius': int(config['inner_radius']),
                    'maxRadius': int(1.2*config['inner_radius'])}

    outercircle = np.empty(3, dtype=int)
    innercircle = np.empty(3, dtype=int)
    outercircle = get_circle(intimage, 200, params_big)
    innercircle = get_circle(intimage, 300, params_small)

    if do_plot == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for (x, y, r) in outercircle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(intimage, (x, y), r, (128, 128, 128), 4)
            cv2.rectangle(intimage, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 128), -1)
        ax1.imshow(intimage, origin='lower')
        for (x, y, r) in innercircle:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(intimage, (x, y), r, (128, 128, 128), 4)
            cv2.rectangle(intimage, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 0), -1)
        ax2.imshow(intimage, origin='lower')
        fig.savefig(path+'/detail5.png')
    # print("outer circle is", outercircle, "inner circle is", innercircle, flush=True)
    # print(params_big, params_small, flush=True)
    return outercircle, innercircle


def _getEfdData(client, dataSeries, startTime, endTime):
    import asyncio
    """A synchronous warpper for geting the data from the EFD.
    This exists so that the top level functions don't all have to be async def.
    curtesy of Merlin Levine-Fisher.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(client.select_time_series(dataSeries, ['*'], startTime, endTime))


def getEFDinfo(dayObs, seqNum):
    """ Wrapper that grabs the EFD info for each sequence, currently this
    wrapper does not work, there seems to be an issue with asyncio."""
    from astropy.time import Time, TimeDelta
    from lsst_efd_client import EfdClient
    # import pandas as pd

    client = EfdClient('summit_efd')
    butler = dafPersist.Butler('/project/shared/auxTel/rerun/quickLook')

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

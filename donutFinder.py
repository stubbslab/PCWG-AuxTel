# This is the attempt at a class based update for the donut finding code.
# This should make it possible to better keep track of information, and ease
# future debugging issues
from distutils.log import error
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from lsst.rapid.analysis.imageExaminer import ImageExaminer
import lsst.rapid.analysis.butlerUtils as butlerUtils
from lsst_efd_client import EfdClient
import os
import logging
import math


class DonutFinder():
    def __init__(self, doPlot=False, path=None, butlerLoc='NCSA',
                 efdClient='summit_efd'):
        self.doPlot = doPlot
        if not path:
            self.path = os.path.expanduser('~')
        else:
            self.path = path
        # instantiating the butler and efd clients
        self.butler = butlerUtils.makeDefaultLatissButler(butlerLoc)
        self.efdClient = EfdClient(efdClient)

        # Making a logger
        self.logger = logging.getLogger("DonutFinder")

        # This is the default configuration, which can easily be overwritten
        self.config = {
            "Halfbox": 1200,
            "kernel": 61,
            "minclip": 0.1,
            "outer_radius": 750,
            "inner_radius": 290,
            "vmin": 10,
            "vmax": 500,
            "normPercent": 85,
            "skyPercent": 10,
            "min_dist_outer": 200,
            "min_dist_inner": 300
        }

        # Generating an empty dict, that we can collect results in
        self.results = {}

    def findCircles(self, dataIds, useCutout=False, config=None):
        """Let's find all the circles! this function simply loops over a
        list of exposures (dataIds) the function FindCircle, which does all
        the work for a single exposure.

        Parameters
        ----------
        dataIds : `list`
            list of dataIds of the exposures that we want to find circles for.

        config : `dict`
            Optional configuration dictionary, if not provided we will use a
            standard configuration defined below.

        path : `string`
            Optional path where detail plots should be saved.

        Returns
        -------
        results : `dict`
            Dictionary of all the collected up results, specifically we will
            add results for the efd_infos, the displacements in
            x and y (dxs, dys) and coefficients for the flux skew.
        """
        self.logger.info(f"Starting findCircles for {dataIds}")

        self.results['positions'] = []
        self.results['dxs'] = []
        self.results['dys'] = []
        self.results['coefficients'] = []
        self.results['data_Id'] = []

        if config is not None:
            self.config = config
        self.logger.info(f"running with the following configuration: {self.config}")

        # starting the main loop, running over all sequences given
        for dataId in dataIds:
            outer_circle = np.zeros(3, dtype=int)
            inner_circle = np.zeros(3, dtype=int)

            self.logger.info(f"running algorithm on: {dataId}")

            outer_circle, inner_circle, coef = self.findCircle(dataId, useCutout)

            centration_offset = outer_circle - inner_circle

            position = self.get_efd_info(dataId)
            self.logger.info(f"result: dx_offset={centration_offset[0,0]},dy_offset={centration_offset[0,1]}")
            self.logger.info(f"Coefficients result: {coef}")
            self.results['coefficients'].append(coef[:])
            self.results['dxs'].append(centration_offset[0, 0])
            self.results['dys'].append(centration_offset[0, 1])
            self.results['positions'].append(position)
            self.results['data_Id'].append(dataId)

        return self.results

    def findCircle(self, dataId, useCutout=False, config=None):
        """This function does all the tricks to find the circle for a single
        exposure and returns the inner and outer circles for it.

        Parameters
        ----------
        dataId : `dict`
            Dictionary of Id for the exposure we are attempting to anlyze.

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

        plane_coefficient : `list`
            list of the coefficients (C) needed to plot a plane.
            z = C[0]*x = C[1]*y + C[2]
        """
        if config is not None:
            self.config = config
        self.logger.info(f"running with the following configuration: {self.config}")

        path = os.path.join(self.path, f"detail_plots{dataId['day_obs']}", f"seq{dataId['seq_num']:05}")

        if self.doPlot:
            self.logger.info(f" path for plots folder has been set to: {path}")
            self._pathcheck(path)

        if self.doPlot:
            try:
                os.makedirs(path, exist_ok=True)
            except FileExistsError:
                self.doPlot = False
                print(f"Seems like a file exists at {path}, so we can't make a folder.")
            except PermissionError:
                self.doPlot = False
                print(f"We cannot save the files to {path}, we lack permission.")

        exp = self.butler.get('quickLookExp', dataId)

        if useCutout:
            imexam = self._examine(exp)
            image = self._cutOut(imexam)
        else:
            image = np.array(exp.image.array)

        norm_image, cutout_smoothed = self._smoothNormalized(image)

        int_image = self._detectMask(norm_image)

        params_big = {'param1': 10, 'param2': 10, 'minRadius': int(self.config['outerRadius']),
                      'maxRadius': int(1.2*self.config['outerRadius'])}
        params_small = {'param1': 30, 'param2': 10, 'minRadius': int(self.config['innerRadius']),
                        'maxRadius': int(1.2*self.config['innerRadius'])}

        outer_circle = np.empty(3)
        inner_circle = np.empty(3)
        outer_circle = self._applyHoughTransform(int_image, self.config['minDistOuter'], params_big)
        inner_circle = self._applyHoughTransform(int_image, self.config['minDistInner'], params_small)
        path2 = path

        if self.doPlot:
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

        plane_coefficient = self._findPlaneSkew(norm_image, path2)

        return outer_circle, inner_circle, plane_coefficient

    def _examine(self, exp):
        '''This function has been deprecated. It is only called if we need to
        cutout the donut from a larger set of data.'''
        imexam = ImageExaminer(exp, boxHalfSize=self.config["halfbox"])
        return imexam

    def _cutOut(self, imexam):
        '''This function has been deprecated. It is only called if we need to
        cutout the donut from a larger set of data.'''
        cutout = np.array(imexam.data)
        return cutout

    def _smoothNormalized(self, image):
        kernel = self.config['kernel']
        cutoutSmoothed = cv2.GaussianBlur(image, (kernel, kernel), 0)

        # We normalize and remove background sky values:
        normValue = np.percentile(cutoutSmoothed, self.config['normPercent'])
        skyValue = np.percentile(cutoutSmoothed, self.config['skyPercent'])

        normImage = (cutoutSmoothed - skyValue)/normValue

        return normImage, cutoutSmoothed

    def _detectMask(self, norm_image):
        # Adding the option that we can either use a predefined value,
        # or we can use an automatically calculated clip.
        if 'maxclip' in self.config:
            maxclip = self.config['maxclip']
        else:
            max_img = np.max(norm_image)
            mean_img = np.mean(norm_image)
            maxclip = max_img/2 - mean_img


        normMask = ma.getmask(ma.masked_greater_equal(norm_image, maxclip))
        intImage = np.uint8(255*normMask)

        return intImage

    def _applyHoughTransform(self, intImage, min_dist, params):
        circle = cv2.HoughCircles(intImage, cv2.HOUGH_GRADIENT, 1, min_dist, **params)

        circle = np.round(circle[0, :]).astype(int)
        return circle

    def _findPlaneSkew(self, normImage, path):
        """What if we wanted to also find the skew, of the flux plane? Here we
        attempt to fit a plane to the normalized fluxes, so in case we do see
        that the flux is uneven across the image, we have the coefficients
        for a: z = a*x + b*y + c plane. with z being the flux.
        This implementation is heavily inspired by the example given in
        https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
        """

        nmi, mask = self._makeMask(normImage)

        if self.doPlot:
            path = os.path.join(path, "cutout.png")
            fig = plt.figure()
            plt.imshow(nmi, origin='lower')
            plt.colorbar()
            fig.savefig(path)

        grid = np.indices(nmi.shape)
        y = ma.array(grid[0], mask=mask)
        x = ma.array(grid[1], mask=mask)
        z = ma.array(normImage, mask=mask)
        data = np.c_[x.compressed(), y.compressed(), z.compressed()]
        a = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        import scipy.linalg as linalg
        c, _, _, _ = linalg.lstsq(a, data[:, 2])
        return c

    def _makeMask(self, normImage):
        if 'maxclip' in self.config:
            maxclip = self.config['maxclip']
        else:
            max_img = np.max(normImage)
            mean_img = np.mean(normImage)
            maxclip = max_img/2 - mean_img

        nmi = ma.masked_less_equal(normImage, maxclip)
        mask = ma.getmask(nmi)
        return nmi, mask

    def _pathcheck(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            self.doPlot = True
        except FileExistsError:
            self.doPlot = False
            self.logger.error(f"Seems like a file exists at {path}, so we can't make a folder.")
            self.logger.error(" We have therefore turned off plotting.")
        except PermissionError:
            self.doPlot = False
            self.logger.error(f"We cannot save the files to {path}, we lack permission.")
            self.logger.error(" We have therefore turned off plotting.")

    def _getEfdData(self, dataSeries, startTime, endTime):
        """A synchronous wrapper for geting the data from the EFD.
        This exists so that the top level functions don't all
        have to be async def.
        curtesy of Merlin Levine-Fisher.
        """
        import asyncio
        import nest_asyncio
        # This is the magic that let's us call this asyncio loop from inside a
        # jupyternotebook
        nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.efdClient.select_time_series(dataSeries,
                                                                         ['*'], startTime, endTime))

    def get_efd_info(self, dataId):
        """ Wrapper that grabs the EFD info for each sequence"""

        where = "exposure.day_obs=day_obs AND exposure.seq_num=seq_num"
        expRecords = self.butler.registry.queryDimensionRecords("exposure", where=where,
                                                                bind={'day_obs': dataId['day_obs'],
                                                                      'seq_num': dataId['seq_num']})
        expRecords = list(expRecords)
        assert len(expRecords) == 1, self.logger.error(f'Found more than one exposure record for {dataId}')
        record = expRecords[0]
        t_start = record.timespan.begin
        t_end = record.timespan.end

        hex_position = self._getEfdData("lsst.sal.ATHexapod.positionStatus", t_start.utc, t_end.utc)
        # This dictionary gives the hexapod names and indices
        # units for x,y,z in mm and u,v,w in degrees, according to
        # https://ts-xml.lsst.io/sal_interfaces/ATHexapod.html#positionupdate.

        names = ['x', 'y', 'z', 'u', 'v', 'w']
        positions = {}
        for val, name in enumerate(names):
            key = f'reportedPosition{val}'
            position = hex_position[key][0]
            positions[name] = position

        return positions

# What follows is a new addition to get WFS inversion

    def positions(self, dataIds, focus):
        fig = plt.figure(figsize=(10, 10))

        for dataId in dataIds:
            pos = self.get_efd_info(dataId)
            print(f"Sequence {dataId['seq_num']} has positions x: {pos['x']-focus[0]} y: {pos['y']-focus[1]}")
            plt.scatter(pos['x']-focus[0], pos['y']-focus[1])

        fig.show()

    def WFSinversion(self, dataId_1, dataId_2, focus, config=None):
        ''' WORK IN PROGRESS, we work on 2 images at a time.
        '''

        if config is not None:
            self.config = config
        self.logger.info(f"running with the following configuration: {self.config}")

        # Let's start by grabbing positions:
        pos_1 = self.get_efd_info(dataId_1)
        pos_2 = self.get_efd_info(dataId_2)

        if math.isclose(abs(pos_1['x']-focus[0]), abs(pos_2['x']-focus[0]), rel_tol=0.05):
            self.logger.info("we are comparing along x axis")
        elif math.isclose(abs(pos_1['y']-focus[1]),abs(pos_2['y']-focus[1]), rel_tol=0.05):
            self.logger.info('we are comparing along y axis')
        else:
            self.logger.error(f"The two dataID's {dataId_1} and {dataId_2} are not compatible")
            self.logger.error(f"positions for image 1: ({pos_1['x']-focus[0]},{pos_1['y']-focus[1]})")
            self.logger.error(f"while for image 2: ({pos_2['x']-focus[0]},{pos_2['y']-focus[1]}")
            raise ValueError

        exp_1 = self.butler.get('quickLookExp', dataId_1)
        exp_2 = self.butler.get('quickLookExp', dataId_2)

        image_1 = np.array(exp_1.image.array)
        image_2 = np.array(exp_2.image.array)

        norm_image_1, _ = self._smoothNormalized(image_1)
        norm_image_2, _ = self._smoothNormalized(image_2)

        _, mask_1 = self._makeMask(norm_image_1)
        _, mask_2 = self._makeMask(norm_image_2)

        masked_image_1 = ma.array(image_1, mask=mask_1)
        masked_image_2 = ma.array(image_2, mask=mask_2)

        # Getting the details for the outer circle
        outer_circle_1, _, _ = self.findCircle(dataId_1)
        outer_circle_2, _, _ = self.findCircle(dataId_2)

        # Selecting the larger of the two radii to use for cutting the donuts
        # out with.
        radii = max(outer_circle_1[0][2], outer_circle_2[0][2])

        # Reducing our image to only the donut, centered on
        # the outer circle's center.
        cut_masked_image_1 = masked_image_1[outer_circle_1[0][1]-radii:outer_circle_1[0][1]+radii,
                                            outer_circle_1[0][0]-radii:outer_circle_1[0][0]+radii]
        cut_masked_image_2 = masked_image_2[outer_circle_2[0][1]-radii:outer_circle_2[0][1]+radii,
                                            outer_circle_2[0][0]-radii:outer_circle_2[0][0]+radii]
        cut_mask_2 = mask_2[outer_circle_2[0][1]-radii:outer_circle_2[0][1]+radii,
                            outer_circle_2[0][0]-radii:outer_circle_2[0][0]+radii]
        # flip image 2:
        #cut_masked_image_2.mask = ma.nomask
        flipped_cm_image_2 = np.flip(cut_masked_image_2)
        #flip_mask_2 = np.flip(cut_mask_2)
        #print(cut_mask_2.shape)
        #print(flip_mask_2)
        #print(flip_mask_2.shape, flipped_c_image_2.shape)
        #flipped_cm_image_2 = ma.array(flipped_c_image_2, mask=flip_mask_2)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(cut_masked_image_2, origin='lower')
        axs[0, 0].set_title('not flipped')
        axs[0, 1].imshow(flipped_cm_image_2, origin='lower')
        axs[0, 1].set_title('flipped image')

        difference = cut_masked_image_1 - flipped_cm_image_2
        average = ma.array((cut_masked_image_1, flipped_cm_image_2)).mean(axis=0)
        axs[1, 0].imshow(difference, origin='lower')
        axs[1, 0].set_title('difference')
        axs[1, 1].imshow(average, origin='lower')
        axs[1, 1].set_title('average')
        
        fig.show()

        # Step 6 in Chris's plan
        rel_diff = ma.divide(difference, average)

        summed = ma.sum(rel_diff)

        # Missing step 6.25

        corrected_rel_diff = rel_diff - summed

        fig2, ax2 = plt.subplots(1,2, figsize=(10, 10))
        ax2[0].imshow(rel_diff, origin='lower')
        ax2[1].imshow(corrected_rel_diff, origin='lower')
        ax2[0].set_title('Relative difference')
        ax2[1].set_title('corrected relative difference')

        fig2.show()
        # Missing steps 7.5

        x_tilt = np.trapz(corrected_rel_diff, axis=0)
        y_tilt = np.trapz(corrected_rel_diff, axis=1)

        pixel_tilt = np.zeros_like(corrected_rel_diff)

        for i in range(len(x_tilt)):
            for j in range(len(y_tilt)):
                pixel_tilt[i, j] = np.sqrt(x_tilt[i]**2 + y_tilt[j]**2)

        return x_tilt, y_tilt, pixel_tilt

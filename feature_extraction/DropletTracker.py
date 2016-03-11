import sys, os, time, math, csv, operator
import numpy, cv2
from scipy.ndimage import label
import time, multiprocessing
from glob import glob
import re


class DropletTracker:


    def __init__(self, filename):
        ''' if debug is True you will see a video of the image processing
        otherwise it is done silently (and faster)'''
        self.filename = filename

    def analyse_video(self, debug = False):
        self.total_frames = 0
        self.track_data = self.img_proc(self.filename, debug)

    def img_proc(self, filename, debug):
        '''given a video it will return a list where every position is a frame, and in each frame
        there will be the centres of droplet objects. The droplets are loosely identified because
        we only need ids between frames and not in the whole video. The id are given simply by
        overlapping droplets and assigning the nearest droplet object from the previous frame'''

        # prepare the MOG backgrund substractor
        backsub = cv2.BackgroundSubtractorMOG()
        capture = cv2.VideoCapture(filename)

        # use first frame to describe arena area
        ret, frame = capture.read()
        arena = numpy.zeros((frame.shape[0],frame.shape[1],1), numpy.uint8)

        # using hough searches for a big circle -> the dish
        cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
                cimg, cv2.cv.CV_HOUGH_GRADIENT, 4, 1, param1=30, param2=300, minRadius=200, maxRadius=280)

        if circles is not None:
            dish = circles[0][0] #it returns by order of accumulator, so first is the best
        else: # dish not detected, something failed. It will crash in the next line
            print "hough failed"

        # we define the arena as the dish detected minus 20 pixels radius
        # it will just be a mask, that we will use later to AND with the working frames
        cv2.circle(arena, (dish[0], dish[1]), int(dish[2]-20), 255, -1)

        obj_center = {}
        droplets_ids = 0 # just to give an id to each droplets. it starts with 0 and then +1 for every droplets detected
        track_data = []

        while True:
            ret, frame = capture.read()

            if not ret:
                break

            self.total_frames += 1

            # uses canny first, and then fill flood, to detect closed objects (as droplets)
            canny_segmentation = self.canny_detector(frame, arena)

            # perform MOG backgrund substraction to detect droplets
            fgmask = backsub.apply(frame, None, 0.005)
            # arena is only true in arena's area, so this blackouts everything else
            fgarena = cv2.bitwise_and(fgmask, arena)
            # we erode now because MOG and canny_detector output different sizes
            # MOG returns slightly bigger droplets, so we erode it to make it similar
            # to canny detector
            fgarena = cv2.erode(fgarena, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
            combination = cv2.bitwise_or(fgarena, canny_segmentation)
            contours, _ = cv2.findContours(combination, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            circles, centers = [], []

            for c in contours:
                c = c.astype(numpy.int64)

                if 1 < cv2.contourArea(c) < 2000:
                    circles.append(c)
                    box = cv2.boundingRect(c)
                    centers.append( (box[0] + (box[2] / 2), box[1] + (box[3] / 2)) )

            obj_center, droplets_ids = self.track_droplets(obj_center, centers, droplets_ids)
            track_data.append( obj_center.copy() )

            if debug == True:

                for i in xrange(len(circles)):
                    cv2.drawContours(frame, circles, i, (0, 255, 0))

                # draw the working area in the video
                cv2.circle(frame, (dish[0], dish[1]), int(dish[2]-20), (0,0,255), 3)
                cv2.circle(frame, (dish[0], dish[1]), int(dish[2]), 255, 3)
                cv2.imshow("", frame)
                cv2.waitKey(1)

        return track_data


    def canny_detector(self, frame, arena):
        ''' first use canny to find edges. Then fill the image from 0,0. Everything that is not filled
        is a droplet, because the droplets are the only fully closed objects'''

        # first use canny to detect the edges
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny_res = cv2.Canny(cv2.GaussianBlur(gray_frame, (5, 5), 2) , 30, 60)

        # morpho to join edges from circles which may be broken after canny
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        canny_dilate = cv2.dilate(canny_res, kernel)

        # segmentation based on inner part
        height, width = canny_dilate.shape
        mask = numpy.zeros((height+2, width+2), dtype=numpy.uint8)
        cv2.floodFill(canny_dilate, mask, (0, 0), 255)
        canny_arena = cv2.bitwise_and(canny_dilate, arena)
        canny_arena[canny_arena == 1] = 0
        mask_new = numpy.zeros((height+2, width+2), dtype=numpy.uint8)
        cv2.floodFill(canny_arena, mask_new, (0, 0), 255)

        components = self.segment_on_dt(canny_arena)

        return components


    def segment_on_dt(self, img):
        ''' see: http://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html '''

        border = img - cv2.erode(img, None)

        dt = cv2.distanceTransform(255 - img, cv2.cv.CV_DIST_L2, 3)
        if dt.max() > dt.min():
            dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
        else:
            dt = ((dt - dt.min()) / 1).astype(numpy.uint8)
        _, dt = cv2.threshold(dt, 50, 255, cv2.THRESH_BINARY)

        lbl, ncc = label(dt)
        lbl[border == 255] = ncc + 1

        lbl = lbl.astype(numpy.int32)
        cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
        lbl[lbl < 1] = 0
        lbl[lbl > ncc] = 0

        lbl = lbl.astype(numpy.uint8)
        lbl = cv2.erode(lbl, None)
        lbl[lbl != 0] = 255

        return lbl


    def track_droplets(self, prevcenters, centers, droplets_ids):
        ''' Based just on overlapping between frames
        prevcenters is previous frame, centers is current frame'''

        new_objcenter = {}

        for c in centers:

            # if there were dropletd detected in the previous frame
            if len(prevcenters) > 0:
                # assign to current droplet the nearest in the prev frame
                best = min( (abs(c[0]-oc[0])**2+abs(c[1]-oc[1])**2, i)
                    for i, oc in prevcenters.iteritems() )

            # if the nearest droplets was farer away than 1000 square pixels, ignore it
            if ( len(prevcenters) > 0) and ( best[0] < 1000 ): # 1000 is d threshold
                new_objcenter[best[1]] = c
                del prevcenters[best[1]]

            else: # otherwise we consider it a possible new droplet
                tag = droplets_ids
                droplets_ids += 1
                new_objcenter[tag] = c

        return new_objcenter, droplets_ids


    def ffx_movement(self):
        '''calculates the sum of avgs of droplet movement over frames'''

        results = self.track_data

        total_d = 0.0

        for i in range(len(results)):

            if len(results[i]) > 1:

                for k,v in results[i].iteritems():
                    d = 0.0

                    if k in results[i-1]:
                        v_1 = results[i-1][k]
                        d += math.sqrt( (v[0] - v_1[0])**2 + (v[1] - v_1[1])**2 )

                total_d += float(d) / len(results[i])

        return total_d / self.total_frames


    def ffx_division(self):
        ''' returns number of alive droplets in the last frame'''
        return len(self.track_data[-1])


    def ffx_directionality(self):
        ''' calculates the sum of avgs of droplets changes of directions
        over frames. Formula
        http://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
        '''

        results = self.track_data

        total_c = 0.0

        for i in xrange(len(results)):

            if len(results[i]) > 1: # we force to have at least 2 droplets
                drop_counter = 0

                for k,v_0 in results[i].iteritems():
                    c = 0.0

                    if k in results[i-1] and k in results[i-2]:
                        v_2 = results[i-2][k]
                        v_1 = results[i-1][k]
                        a = ( v_1[1] - v_2[1], v_1[0] - v_2[0] )
                        b = ( v_0[1] - v_1[1], v_0[0] - v_1[0] )
                        dot_prod = sum(map( operator.mul, a, b))
                        denominator = numpy.linalg.norm(a)*numpy.linalg.norm(b)
                        if denominator != 0:
                            div = dot_prod / denominator
                            if div > 1: div = 1 # stupid float problems
                            if div < -1: div = -1
                            c += math.acos( div )
                        drop_counter += 1

                if drop_counter > 0:
                    total_c += float(c) / drop_counter

        return math.degrees(total_c / self.total_frames)


if __name__ == "__main__":

    analysis = DropletTracker(sys.argv[1], debug = False)
    print analysis.ffx_movement()
    print analysis.ffx_division()
    print analysis.ffx_directionality()

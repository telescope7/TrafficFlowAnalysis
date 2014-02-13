Traffic Flow Analyzer is a program designed to detect and measure uni-directional objects in videos.

The software works by loading the specified video, applying a background subtraction if specified, blurring the frame by a specified amount, thresholding the frame by a specified amount, running a contour detection algorithm
and selecting objects that surpass the minimum object diameter requirement.  These objects will be added to the Moving Object database if not present.  Otherwise, the object is compared to other previously tracked objects
and mapped to the appropriate moving instance based on forward movement (using either a look ahead window or an overlapping object boundry analysis).  Once the object is no longer tracked or exits the field of view, the 
statistics and measurements of the tracked object are reported.  Each tracked object will be recorded on STDOUT.  The objects measurements include:

* Number of frames tracked
* First X coordinate 
* First Y coordinate
* Last X coordinate
* Last Y coordinate
* First frame tracked
* Last frame tracked
* Average radius of enclosing circle
* Average width of enclosing contour
* Average heigh of enclosing contour
* Average area of enclosing contour

---

Table of Contents

* <a href="#How to use">How to use</a>
* <a href="#Contact">Contact</a>

---


<a name="How to use"></a>

# How to use

./traffic_analyzer.py [OPTIONS]

'-d', '--debug', debug mode

'-m', '--movie', movie.avi

'-b', '--blur', blur setting

'-t', '--threshold', threshold

'-c', '--objectdiametermin', object minimum diameter

'-x', '--boxwindow', box window size

'-p', '--playbackspeed'  playback speed

'-v', '--visualize', visualize tracking

'-s', '--subtractbg', subtract background

'-a', '--accumulator', accumulator weight

'-o', '--output', output avi

Example:

    ./traffic_analyzer.py -c 315 -p 35 -b 45 -t 30 -s -m videos/traffic.avi -v


<a name="Contact"></a>

# Contact

Traffic Flow Analyzer was developed by [Matthew Thomas](https://github.com/telescope7/) 


Traffic Flow Analyzer is a program designed to detect and measure uni-directional objects in videos.

Each tracked object will be recorded on STDOUT.  The objects measurements include:

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

* <a href="#Contact">Contact</a>
* <a href="#How to use">How to use</a>

---

<a name="Contact"></a>

# Contact

Traffic Flow Analyzer was developed by [Matthew Thomas](https://github.com/telescope7/) 


<a name="How to use"></a>

# How to use

./traffic_analyzer.py [OPTIONS]

'-d', '--debug', action='store_true',default=False, help="debug mode"

'-m', '--movie', help='movie.avi'

'-b', '--blur', help='blur setting'

'-t', '--threshold', help='threshold'

'-c', '--objectdiametermin', help='object minimum diameter'

'-x', '--boxwindow', help='box window size'

'-p', '--playbackspeed', help='playback speed'

'-v', '--visualize', action='store_true',  help='visualize tracking'

'-s', '--subtractbg', action='store_true',  help='subtract background'

'-a', '--accumulator', help='accumulator weight'

'-o', '--output', help='output avi'

Example:

    ./traffic_analyzer.py -c 315 -p 35 -b 45 -t 30 -s -m videos/traffic.avi -v
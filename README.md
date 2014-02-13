Traffic Flow Analyzer is a program designed to detect and measure uni-directional objects.  

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
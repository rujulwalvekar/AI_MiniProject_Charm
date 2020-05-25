

from flask import Response,Flask,render_template


app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

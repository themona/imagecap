from flask import Flask , render_template ,redirect,request
import caption_it
app = Flask(__name__)



@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/', methods=['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)
		f.save(path)
		caption = caption_it.caption_this_image(path)
		print(caption)

		result_dic = {'image': path , 'gen_caption': caption}
	return render_template("index.html", results = result_dic )


if __name__ == '__main__':
	#app.run()
	app.run(debug= True)
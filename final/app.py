from flask import Flask,jsonify,redirect,abort,make_response,request,render_template,url_for,Response
import os
from werkzeug.utils import secure_filename
#import face
import uuid
import api
import usersDb

UPLOAD_FOLDER = os.path.join(os.getcwd(),'celebs')
VIDEO_FOLDER = os.path.join(os.getcwd(),'images')
db = usersDb.UsersDb()
FACES_FOLDER = os.path.join(os.getcwd(),'faces')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4','avi'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER
app.config['IMAGE_FOLDER'] = VIDEO_FOLDER


import cv2
import face_detect
from predictor import predict

coords = False
# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

camera = cv2.VideoCapture(0)
def gen_frames():  # generate frame by frame from camera
    w=0;h=0
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            # print(f"{frame.shape}")
            # print(coords)
            for (x,y,w,h) in face_detect.getFacesDLIB(frame):
                # print(x,x+w,y,y+h)
                # print(coords)
                # predict(frame[y:y+h,x:x+w])

                # if coords and (x-20 <= coords[0] <= x+w+20) and (y-20 <= coords[1] <= y+h+20):
                    # cv2.putText(frame, 'OpenCV', (x,y), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
            # print(f"width:{w}, height:{h}")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/getInfo")
def getInfo():
    global coords
    
    
    print("BOOOOOOM")
    x = request.args.get("x")
    y = request.args.get("y")
    print("BOOOOM")

    ok, frame = camera.read()
    if ok:
        lcoords = (eval(x),eval(y))
        for (x,y,w,h),encoding in face_detect.getFacesDLIB(frame, True):
            print("BOOOOM")
            print(x,x+w,y,y+h)
            print(lcoords)
            # age, gender = predict(frame[y:y+h,x:x+w])


            if lcoords and (x-20 <= lcoords[0] <= x+w+20) and (y-20 <= lcoords[1] <= y+h+20):
                face = frame[y-20:y+h+20,x-20:x+w+20]
                age,gender = predict(face)
                name = api.predict_from_encoding(encoding)
                user_info = db.getUser(name) 
                coords = lcoords
                return jsonify({'prediction':{'name': name,'age': f"{age[0][0]}",'gender':gender},
                                 'found':1, 'actual':user_info})
    return jsonify({'res':"No faces found", 'found':0  })  

def send_message(number, msg):
    url = "https://www.fast2sms.com/dev/bulk"
    querystring={"authorization":"SdmgeRaqKu054Zbst2EpvHfcVD13rGQnkLwxjXNB98U7JPToYiI34KEaD9bPpi8woFlzQAYWfVyrNB02","sender_id":"FSTSMS","message":"This is test message","language":"english","route":"p","numbers":f"{number}"}
    headers = {
    'cache-control': "no-cache"
	}
    response = requests.request("GET", url, headers=headers, params=querystring)
    print(response.text)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/live")
def index():
    return render_template("live.html")

@app.route("/elements")
def elements():
    return render_template("elements.html")

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        name = request.form['name']
        print(name,"INSIDE")
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        db.addUser({
            'name':name,
            'age':age,
            'gender':gender,
            'phone':phone,
        })
        db.save()

        file = request.files.get('image')
        celebrities = os.listdir(app.config['UPLOAD_FOLDER'])
        if name.strip() not in celebrities:
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],name.strip()))
                os.makedirs(os.path.join(app.config['FACES_FOLDER'],name.strip()))


        if file:
            filename = secure_filename(file.filename);print("BOOM")
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], name.strip(),str(uuid.uuid1())
+"."+filename.split('.')[1]);print(img_path)
            file.save(img_path)
            api.train(img_path,name.strip());print("BOOM")
            return redirect(url_for('elements',msg="success"))
    else:
        return render_template("elements.html")


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            print("hello")
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['IMAGE_FOLDER'],str(uuid.uuid1())
+"."+filename.split('.')[1])
            file.save(image_path)
            image,prediction = api.predict(image_path)
            return render_template("elements.html", image=image)


        #     print(image)
        #     return render_template("test.html",image=image)
    else:
        print(name,"INSIDE");return render_template("test.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            print("hello")
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['IMAGE_FOLDER'],str(uuid.uuid1())
+"."+filename.split('.')[1])
            file.save(image_path)
            image,prediction = api.predict(image_path)
            return jsonify({'prediction':prediction})


        #     print(image)
        #     return render_template("test.html",image=image)
    else:
        return render_template("test.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)

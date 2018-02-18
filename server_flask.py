from test_dogs import CVModel as DogModel, main as dog_main, results as dog_results
#from salesforce import SfObject
#from test import main, results as test_results, CVModel
import requests
from pylons import response
import json,os
import time
import random
from string import Template
import threading
from flask import Flask, request, redirect, url_for, Response
from werkzeug.utils import secure_filename


my_template = Template("Hello, ${person_name}, how are you?")
app = Flask(__name__)
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
queue = []
@app.route('/send_image', methods=['GET', 'POST'])
def process_file():
    confidence = 0
    if request.method =='GET':
        return "Please post an image for Computer vision to execute"
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = str(int(time.time()))+"_"+str(random.randint(1,1000))+"_"+filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = main(os.path.join(app.config['UPLOAD_FOLDER'], filename),via_messenger=False)
            # if not  request.headers.get('User-Agent'):
            #     return Response(json.dumps({"response": resp,"file_id":filename}, ensure_ascii=False), mimetype='application/json')
            # else:
            #     image_id = Template("image ID is - '$image_id'<br>")
            #     class_type = Template("image class is - '$image_class'<br>")
            #     thumbs_up = Template('<button id="thumbsup"></button><script>document.getElementById("thumbsup").html = "up";</script>')
            #     thumbs_down = Template('<button id="thumbsdown"></button><script>document.getElementById("thumbsdown").html = "down";</script>')
            #     return image_id.substitute({'image_id':filename}) +class_type.substitute({'image_class':resp}) + thumbs_down.template + thumbs_up.template
            return Response(json.dumps({"class_type": resp, "file_id": filename}, ensure_ascii=False),
                        mimetype='application/json')


@app.route("/thumbsup",methods=['GET'])
def thumbsup():
    if request.method =='GET':
        image_id = request.args.get('file_id')
        append_to_file(image_id,"YES")
    return "Thank you for your feedback"

@app.route("/thumbsdown",methods=['GET'])
def thumbsdown():
    if request.method =='GET':
        image_id = request.args.get('file_id')
        append_to_file(image_id,"NO")
    return "Thank you for your feedback"
@app.route('/upload_image')
def upload_file():
   return '''<html>
   <body>

      <form action = "http://45.55.197.141:8039/send_image" method = "POST"
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>

   </body>
</html>'''

def download_image(url):
    path = 'images/'
    path = 'images/'
    full_path = path+ str(int(time.time()))+"_"+str(random.randint(1,1000))+"_"+url.split("?")[0].split("/")[-1]
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(full_path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return full_path
    else:
        return None
def slack_bot_message(fid,confidence,dog,name):
    fid = str(fid)
    return {
            "text": "Product Recognition Bot could not recognize the product. Please assist.",
            "attachments": [
                {
                    "text": "Engine was {0}% sure that it was a {1} dog".format(confidence,dog),
                    "fallback": "Are you able to make selections",
                    "callback_id": "wopr_game",
                    "color": "#3AA3E3",
                    "attachment_type": "default",
                    "image_url": None,
                    "actions": [
                        {
                            "name": "dog",
                            "text": "chihuahua",
                            "type": "button",
                            "value": "chihuahua_"+fid+"_"+name
                        },
                        {
                            "name": "dog",
                            "text": "german shepherd",
                            "type": "button",
                            "value": "german shepherd"+fid+"_"+name
                        },
                        {
                            "name": "no dog",
                            "text": "golden retriever",
                            "type": "button",
                            "value": "golden retriever"+fid+"_"+name
                        },
                        {
                            "name": "dog",
                            "text": "labrador retriever",
                            "type": "button",
                            "value": "labrador retriever"+fid+"_"+name
                        },
                        {
                            "name": "dog",
                            "text": "muffin",
                            "type": "button",
                            "value": "muffin"+fid+"_"+name
                        },
                        {
                            "name": "dog",
                            "text": "pug",
                            "type": "button",
                            "value": "pug"+fid+"_"+name
                        },
                        {
                            "name": "dog",
                            "text": "siberian husky",
                            "type": "button",
                            "value": "siberian husky"+fid+"_"+name
                        }
                    ]
                }
            ]
        }
def append_to_file(file_id,feedback):
    path = 'images/'
    path = 'feedback.txt'
    try:
        with open(path,'a') as f:
            f.write(str(file_id)+" , "+feedback+"\n")
    except Exception as e:
        time.sleep(1)
        with open(path,'a') as f:
            f.write(str(file_id)+" , "+feedback+"\n")

'''
@app.route("/post_image_bulb",methods=["GET"])
def process_image_bulb():
    url = request.url.split("?url=")[1].replace("%2F","/").replace("%3F","?")
    fid = request.args.get('user_id')
    if not url:
        return Response(json.dumps({"response":"There was some problem with your image"},ensure_ascii=False),mimetype='application/json')
    else:
        local_path = download_image(url)
        resp = bulb_main(fid,local_path)
        while not bulb_results.get(fid,None):
            time.sleep(1)
        dog_type,confidence = bulb_results.get(fid,None)[0]
        del bulb_results[fid]
        return Response(json.dumps({"response":dog_type},ensure_ascii=False),mimetype='application/json')


@app.route("/post_image_bag",methods=["GET"])
def process_image_bag():
    url = request.url.split("?url=")[1].replace("%2F","/").replace("%3F","?")
    fid = request.args.get('user_id')
    if not url:
        return Response(json.dumps({"response":"There was some problem with your image"},ensure_ascii=False),mimetype='application/json')
    else:
        local_path = download_image(url)
        resp = dog_main(fid,local_path)
        while not dog_results.get(fid,None):
            time.sleep(1)
        dog_type,confidence = bag_results.get(fid,None)[0]
        del bag_results[fid]
        return Response(json.dumps({"response":dog_type},ensure_ascii=False),mimetype='application/json')
'''

@app.route("/post_image",methods=["GET"])
def process_image():
    confidence = 0
    post_to_slack,post_to_sf = True,True
    url = request.url.split("?url=")[1].replace("%2F","/").replace("%3F","?")
    fid = request.args.get('user_id')
    user_name = request.args.get("user_name")
    if not url:
        return Response(json.dumps({"response":"There was some problem with your image"},ensure_ascii=False),mimetype='application/json')
    else:
        local_path = download_image(url)
        resp = dog_main(fid,local_path)
        while not test_results.get(fid,None):
            time.sleep(1)
        dog_type,confidence = test_results.get(fid,None)[0]
        del test_results[fid]
        if confidence < 80:
            message = slack_bot_message(fid,confidence,dog_type,user_name)
            message['attachments'][0]['image_url'] = url
            slack_hook_url = "https://hooks.slack.com/services/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"
            #slack_hook_url = 'https://hooks.slack.com/services/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'
            if post_to_slack:
                response = requests.post(slack_hook_url,json=message)
            if post_to_sf:
                response = SfObject().create_case(url,fid,user_name,confidence,dog_type)
            return Response(json.dumps({"response":"Manual validation"}),mimetype='application/json')


        #local_path = '/Users/mapss/Downloads/nlp1.jpg'

        return Response(json.dumps({"response":dog_type},ensure_ascii=False),mimetype='application/json')



@app.route("/slack_response",methods=["GET","POST"])
def slack():
    di = dict(request.form)
    di = json.loads(di['payload'][0])
    actions = di['actions'][0]['value']
    dog_type,fid,name = actions.split("_")
    url = di['original_message']['attachments'][0]['image_url']
    queue.append(("slack",(dog_type,fid,url)))
    return "Response sent to "+name

@app.route("/salesforce_response",methods=["GET","POST"])
def salesforce_Response():
    if request.method =='GET':
        comment = request.args.get('comment')
        case_id = request.args.get('case_id')
        fid = request.args.get('bs_user_id')
        print comment
        print case_id
        print fid
        queue.append(("sf", (comment, fid, case_id)))
        return "Response received"

def slack_post_to_messenger(url=None,fid=None,dog=None):
    bot_url = "https://dev.motherbot.co/danfossmodel3/custom/image_manual_response"
    response = requests.post(bot_url,json={"image_url":url,"dog_type":dog,"user_id":fid})
    print response
    return response

def slack_post_to_messenger(url=None,fid=None,dog=None):
    bot_url = "https://dev.motherbot.co/danfossmodel3/custom/image_manual_response"
    response = requests.post(bot_url,json={"image_url":url,"dog_type":dog,"user_id":fid})
    print response
    return response

def salesforce_post_to_messenger(comment,fid=None,caseid=None):
    bot_url = "https://dev.motherbot.co/danfossmodel3/custom/image_manual_response"
    response = requests.post(bot_url,json={"dog_type":comment,"url":fid,"user_id":fid})
    print response
    return response

class SlackResponse(threading.Thread):
    def run(self):
        while True:
            if len(queue)>0:
                typ,tpl = queue.pop()
                if typ == 'slack':
                    dog, fid, url = tpl
                    response = slack_post_to_messenger(url,fid,dog)
                else:
                    comment,fid,case_id = tpl
                    print "posting sf to bot"
                    response = salesforce_post_to_messenger(comment,fid,case_id)
            else:
                time.sleep(1)

if __name__ == "__main__":
    try:
        #obj = CVModel()
        #obj.start()
        obj1 = DogModel()
        obj1.start()
        slack_response = SlackResponse()
        slack_response.start()
        port = 8889
        app.run(host='0.0.0.0',port=int(port))
        #download_image("https://scontent-frt3-1.xx.fbcdn.net/v/t35.0-12/21104175_10155711206998980_563474967_o.png?_nc_ad=z-m&oh=d4823a9ecaf5a095503d9c43cb7401d5&oe=59A18C8D")

    except Exception as e:
        print e
                                         

from flask import Flask,request,render_template,url_for,jsonify
#from flair.models import TARSClassifier
#classifier2 = TARSClassifier.load('tars-base')
#from flair.data import Sentence
from googletrans import Translator
import pickle
clf=pickle.load(open('nlp_model1.pkl','rb'))
cv=pickle.load(open('transform1.pkl','rb'))

app=Flask(__name__)


def f(textt):
  translator = Translator()
  try:

    m=translator.translate(textt,dest="en")
    return(m.text)
  except:
    return textt
'''
def fp(doc):
  sentence = Sentence(doc)

  classes = ["order related","refund issue","arrival issue","NONE","order cancellation issue","damaged product issue","delivery issue","reschedule","send"]

  classifier2.predict_zero_shot(sentence, classes)
  p=sentence

  try:
    return p.tag
  except:
    return "Generall"
    
'''


@app.route('/')
def home():
    
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    
    
    
    if request.method=='POST':
        message=request.form['message']
        fr=f(message)
        data=[fr]
        #vect=cv.transform(data).toarray()
        vect=cv.transform(data)
        #vect=vect.astype('float32')
        my=clf.predict(vect)
    #return render_template("home.html",pre="It belongs to" +" "+ str(my[0])+" "+"class")
        res={
        "message":message,
        "category":str(my[0])
        }
    return res
    



if __name__=='__main__':
    app.run(debug=True)
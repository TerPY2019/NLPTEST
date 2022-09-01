from lib2to3.pgen2 import token
from flask import Flask,render_template,request,Markup
app = Flask(__name__)
import os
import test
import spacy
from spacy import displacy
@app.route('/')
def hello():
    return render_template('homepace.html')

paths = os.getcwd()+"\savefile"
@app.route('/conver_nlp',methods=["post"])
def method_name():
    namefile = request.files.getlist("textfire[]")
    stext = request.form.get('stext')
    tatalfile = []
    for i in namefile:
        joinpath = os.path.join(paths,i.filename)
        i.save(joinpath)
        tatalfile.append(joinpath)
    textnlp = test.bow(tatalfile)
    addtoken = textnlp.ctoken()
    strtext = textnlp.dicttext(stext)
    topbow = textnlp.topfivebow()
    toptfidf = textnlp.topfivetfidf()
    return render_template('homepace.html',show=addtoken,setext=strtext,bowtopfive=topbow,tfidftopfive=toptfidf)

@app.route('/spacytext',methods=["post","get"])
def spacycode():
    nametext = request.form.get('spatext')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(nametext)
    showNER = displacy.render(doc,style="ent")
    return render_template('homepace.html',nershow=Markup(showNER))

if __name__ == '__main__':
    app.run(debug=True,host="192.168.56.1",port="80")
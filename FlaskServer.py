import importlib
gan=importlib.import_module("3dgan_mit_biasfree")

import json
from flask import Flask,url_for
import os





app = Flask(__name__)

@app.route('/')
def hello_world():
    return "GAN Server Started"

result_generater = gan.ganGetOneResult("./ServerModel/biasfree_9610.cptk")
#test: http://127.0.0.1:5000/gan/22%2622%260%2642%2642%2662
@app.route('/gan/<int:minx>&<int:miny>&<int:minz>&<int:maxx>&<int:maxy>&<int:maxz>')
def RequestGANGen(minx,miny,minz,maxx,maxy,maxz):

    gan.SetBoundState(minx, miny, minz, maxx, maxy, maxz)
    result_list=next(result_generater)
    result_list=result_list.ravel()
    back=""
    for b in result_list:
        if b == True:
            back=back+"1"
        else:
            back=back+"0"
    return back

RequestGANGen(0,0,0,30,30,30)
with app.test_request_context():

    print(url_for('RequestGANGen', minx='0', miny='0', minz='0', maxx='30', maxy='30', maxz='30'))

if __name__ == '__main__':
    #app.debug=True
    app.run(host='0.0.0.0')


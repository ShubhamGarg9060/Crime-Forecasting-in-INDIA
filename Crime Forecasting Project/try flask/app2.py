from flask import Flask,render_template,url_for,request,redirect, make_response, jsonify
import random
import json
import pickle
from time import time
from random import random
import numpy as np
from flask import Flask, render_template, make_response
app = Flask(__name__, template_folder='templates')
#from flask_packet import getPacketsPersecond

@app.route('/',  methods=["GET", "POST"])
def main():
    return render_template('aalogin.html')

@app.route('/map',  methods=["GET", "POST"])
def map():
    return render_template('map.html')

@app.route('/andhra',  methods=["GET", "POST"])
def andhra():
    return render_template('andhra.html')

@app.route('/arunachal',  methods=["GET", "POST"])
def arunachal():
    return render_template('arunachal.html')

@app.route('/assam',  methods=["GET", "POST"])
def assam():
    return render_template('assam.html')

@app.route('/bihar',  methods=["GET", "POST"])
def bihar():
    return render_template('bihar.html')

@app.route('/chattisgarh',  methods=["GET", "POST"])
def chattisgarh():
    return render_template('chattisgarh.html')

@app.route('/goa',  methods=["GET", "POST"])
def goa():
    return render_template('goa.html')

@app.route('/guj',  methods=["GET", "POST"])
def guj():
    return render_template('guj.html')

@app.route('/haryana',  methods=["GET", "POST"])
def haryana():
    return render_template('haryana.html')

@app.route('/himachal',  methods=["GET", "POST"])
def himachal():
    return render_template('himachal.html')

@app.route('/jharkhand',  methods=["GET", "POST"])
def jharkhand():
    return render_template('jharkhand.html')

@app.route('/jk',  methods=["GET", "POST"])
def jk():
    return render_template('jk.html')

@app.route('/karnataka',  methods=["GET", "POST"])
def karnataka():
    return render_template('karnataka.html')

@app.route('/kerala',  methods=["GET", "POST"])
def kerala():
    return render_template('kerala.html')

@app.route('/mah',  methods=["GET", "POST"])
def mah():
    return render_template('mah.html')

@app.route('/meghalaya',  methods=["GET", "POST"])
def meghalaya():
    return render_template('meghalaya.html')

@app.route('/mizoram',  methods=["GET", "POST"])
def mizoram():
    return render_template('mizoram.html')

@app.route('/mp',  methods=["GET", "POST"])
def mp():
    return render_template('mp.html')

@app.route('/nagaland',  methods=["GET", "POST"])
def nagaland():
    return render_template('nagaland.html')

@app.route('/odisha',  methods=["GET", "POST"])
def odisha():
    return render_template('odisha.html')

@app.route('/punjab',  methods=["GET", "POST"])
def punjab():
    return render_template('punjab.html')

@app.route('/rajasthan',  methods=["GET", "POST"])
def rajasthan():
    return render_template('rajasthan.html')

@app.route('/sikkim',  methods=["GET", "POST"])
def sikkim():
    return render_template('sikkim.html')

@app.route('/tamilnadu',  methods=["GET", "POST"])
def tamilnadu():
    return render_template('tamilnadu.html')

@app.route('/tripura',  methods=["GET", "POST"])
def tripura():
    return render_template('tripura.html')

@app.route('/up',  methods=["GET", "POST"])
def up():
    return render_template('up.html')

@app.route('/uttarakhand',  methods=["GET", "POST"])
def uttarakhand():
    return render_template('uttarakhand.html')

@app.route('/westbengal',  methods=["GET", "POST"])
def westbengal():
    return render_template('westbengal.html')

@app.route('/zandaman',  methods=["GET", "POST"])
def zandaman():
    return render_template('zandaman.html')

@app.route('/zchandigarh',  methods=["GET", "POST"])
def zchandigarh():
    return render_template('zchandigarh.html')

@app.route('/zdamamdiu',  methods=["GET", "POST"])
def zdamamdiu():
    return render_template('zdamamdiu.html')

@app.route('/zdelhi',  methods=["GET", "POST"])
def zdelhi():
    return render_template('zdelhi.html')

@app.route('/zdnhaveli',  methods=["GET", "POST"])
def zdnhaveli():
    return render_template('zdnhaveli.html')

@app.route('/zlakshadweep',  methods=["GET", "POST"])
def zlakshadweep():
    return render_template('zlakshadweep.html')

@app.route('/zpuducherry',  methods=["GET", "POST"])
def zpuducherry():
    return render_template('zpuducherry.html')

if __name__ == "__main__":
    app.run(debug=True)
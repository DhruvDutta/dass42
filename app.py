from encodings import utf_8
from statistics import mode
from flask import Flask,render_template, request,Response
import json,base64,numpy
import ast,joblib
app=Flask(__name__)

@app.route('/',methods=['GET', 'POST'])

def index():
    model = joblib.load("static/model/Dep_compressed.joblib")
    scale = {'Depression': [(0, 10), (10, 14), (14, 21), (21, 28),(28,100)],
             'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20),(20,100)],
             'Stress': [(0, 15), (15, 19), (19, 26), (26, 34),(34,100)]}
    score=['Normal','Mild','Moderate','Severe','Extremely Severe']
    if request.method == "POST":
        data=request.data.decode('utf-8').split('&')
        if len(data)!=42:
            return Response({'error','Wrong Data'})
        ans_list=numpy.array([ float(i.split('=')[-1]) for i in data ]).reshape(-1,42)
        pred = (model.predict(ans_list)*56).round(2)
        pred_score=[]
        for i in range(3):
            for j in scale[list(scale.keys())[i]]:
                if int(pred[0][i]) in range(j[0],j[1]):
                    pred_score.append(score[scale[list(scale.keys())[i]].index(j)])
        response = {'dep':[int(pred[0][0]),pred_score[0]],
                    'anx':[int(pred[0][1]),pred_score[1]],
                    'strs':[int(pred[0][2]),pred_score[2]],
                    }
        return Response(json.dumps( response ))
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
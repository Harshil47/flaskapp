from flask import flask,render_template
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


# Load the Logistic Regression model
filename = 'nlpsig_clf.pickle'
sig_clf = pickle.load(open(filename, 'rb'))

datas = pd.read_csv('small_final_features.csv')
qdata=pd.read_csv('small.csv')
XS = datas[:20][:]
XS.drop(['id','is_duplicate'], axis=1, inplace=True)
qdata.drop(['id','is_duplicate','question2','qid1','qid2'], axis=1, inplace=True)
arrv=[]
arrq=[]
for i in range(0,16):
    chechlxs=XS[i:i+1]
    sqonlsmall = sig_clf.predict_proba(chechlxs)
    arrv.append(sqonlsmall[0][1])
    if sqonlsmall[0][1] > 0.50:
        
        arrq.append(qdata[i:i+1])
arr = [arrv,arrq]
print(arr)
app = Flask(__name__)  
 
@app.route('/')  
def message():  
      return render_template('message.html',arr=arr , n= len(arr[1]))  
if __name__ == '__main__':  
   app.run(debug = True)   

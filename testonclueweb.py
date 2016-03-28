from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from svmtrain import SvmTrainSGD, SvmTrainGD
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
#......
# Comparison against the text classification from scikit SGDClassifier
# on Clueweb, with Trec web track qrel
#
# following functions are required to run
# pyspark
# relgradedqrel: read in trec web track qrel
# readTfidfVec: read in the document tfidf representation
#......

def predict(X_train, y_train, X_test):
    model = SGDClassifier(l1_ratio=0, fit_intercept=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.decision_function(X_test)
    return y_pred, y_prob
def evalpredict(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    return f1, precision, recall

qids=range(251, 301)
trainrate = 0.6
testrate = 1 - trainrate
# svmtrain = SvmTrainSGD(alpha=0.0001, iteration=5, class_weight=None, shuffle=True)
svmtrain = SvmTrainGD(alpha=0.0001, iteration=100, class_weight=None)

f1diffs = list()
for qid in qids:
    cwidlabel,nonrelcwids = relgradedqrel(sc, qid, qrelf)
    cwidVecs = readTfidfVec(sc, qid, cwidTfidfDir)
    cwids = list()
    ys = list()
    X = list()
    for cwid in cwidVecs.keys():
        if cwid not in cwidlabel and cwid not in nonrelcwids:
            continue
        cwids.append(cwid)
        ys.append(1 if cwid in cwidlabel else 0)
        X.append(cwidVecs[cwid])
    vectorizer=DictVectorizer(sparse=False)
    X=vectorizer.fit_transform(X)
    normX = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(normX, ys, test_size=testrate, stratify=ys)      
    y_pred, y_prob = predict(X_train, y_train, X_test)
    skeval = evalpredict(y_test, y_pred)
    w  = svmtrain.train(X_train, y_train)
    myeval = svmtrain.evalweight(X_test, w, y_test)
    if skeval[0] != 0:
        f1diffs.append((myeval[0] - skeval[0])/ skeval[0])
    print(qid, len(cwidlabel), len(nonrelcwids), len(cwidVecs), myeval[0], skeval[0])
print(sum(f1diffs)/len(f1diffs))

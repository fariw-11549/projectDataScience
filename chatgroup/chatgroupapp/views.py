from django.shortcuts import render
from django.http import HttpResponse
from joblib import load

from sklearn.datasets import fetch_20newsgroups

categories = {
        'sci.space':'บทสนทนาเกี่ยวกับอวกาศ',
        'rec.motorcycles':'บทสนทนาเกี่ยวกับรถจักยานยนต์',
        'misc.forsale':'บทสนทนาเกี่ยวกับการซื้อขาย',
    }
train = fetch_20newsgroups(subset='train', categories=categories)

def index(req):
    result = 'กลุ่มบทสนา ?'
    if req.method == 'POST':
        inp = str(req.POST['input'])
        model = load('./chatgroupapp/static/chatgroup.model')
        pred = model.predict([inp])
        results = train.target_names[pred[0]]
        result = str(categories[results])
        #print(categories[result])
        #print(type(result))

    else:
        result = ''
        print("no")

    return render(req, 'chatgroupapp/chatgroup.html', {
        'result': result
    })


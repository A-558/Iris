import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={'sepal_length':5.1, 'sepal_width':3.5, 'petal_length':1.4 ,'petal_width' : 0.2 })

print(r.json())
import matplotlib.pyplot as plt
import json

with open('model/avgQMax') as f:
	my_list = [json.loads(line) for line in f]

valsAvg = []
for i in my_list:
	valsAvg.append(i['avgQ'])

plt.xlabel('Epochs')
plt.ylabel('Avg QMax')
plt.plot(valsAvg)
plt.savefig('graph/avgQMax.png')
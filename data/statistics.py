import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np


datafile = '/home/yuki/Documents/ymx/stock_research/docs/stock_selection.md'
with open(datafile, 'r') as fd:
    lines = fd.readlines()[10: ]
statdict = {
    'industry': [],
    'prob': []
}
for ln in lines:
    contents = ln[: -1].split(' | ')
    statdict['industry'].append(contents[3])
    statdict['prob'].append(float(contents[-1][1: -1]))
print(f'len(statdict)={len(statdict)}, len(industry)={len(statdict["industry"])}')



def gen_stats(sdata):
    res = OrderedDict()
    for prob, ind in zip(sdata['prob'], sdata['industry']):
        if ind not in res.keys():
            # [total_num, small, middle, large]
            res[ind] = [0, 0, 0, 0]
        res[ind][0] += 1
        if prob < 0.95:
            res[ind][1] += 1
        elif prob > 0.95 and prob < 0.98:
            res[ind][2] += 1
        else:
            res[ind][3] += 1
    res_sorted = sorted(list(res.items()), key=lambda x: x[1][0])
    tick_values = [item[1] for item in res_sorted]
    tick_labels = [item[0] for item in res_sorted]
    return tick_labels, tick_values


tick_labels, tick_values = gen_stats(statdict)
small_segment = [val[1] for val in tick_values]
middle_segment = [val[2] for val in tick_values]
small_middle = [x + y for x, y in zip(small_segment, middle_segment)]
large_segment = [val[3] for val in tick_values]

plt.close()
plt.style.use('ggplot')
# wrong output with chinese characters as tick_label
# solution from https://www.zhihu.com/question/25404709
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 24))
plt.barh(tick_labels, small_segment,height=0.75)
plt.barh(tick_labels, middle_segment, height=0.75, left=small_segment)
plt.barh(tick_labels, large_segment, height=0.75, left=small_middle)
plt.legend(['0.8 < p < 0.95', '0.95 < p < 0.98', '0.98 < p < 1.0'], loc='lower right')
plt.xlabel('frequency')
plt.title('Industry')
plt.savefig('/home/yuki/Documents/ymx/stock_research/docs/images/stock_selection_stats.png', format='png', dpi=200)
plt.show()

from pickle import dump, load
import pandas as pd
import xlwt
from xlwt import Workbook

all_features = load(open("features.p","rb"))


print(type(all_features))
print(len(all_features))

resized_features = {}
# print(all_features)
for key in all_features:
    # print(key, '->', all_features[key][0])
    resized_features[key] = all_features[key][0]

df = pd.DataFrame.from_dict(resized_features, orient="index")

# wb = Workbook()
#
# sheet1 = wb.add_sheet('Sheet 1',cell_overwrite_ok=True)
# sheet1.write(0,0,'IMAGE NAME')
# for i in range(1,256):
#     sheet1.write(0,i,'FEATURE_'+str(i))
#
# wb.save('output.xls')
# #
# df.to_excel("output.xlsx",sheet_name='Sheet_name_1')
# resized_features = {}
# i=1

#   


compression_opts = dict(method='zip',archive_name='out.csv')

df.to_csv('out.zip', index=False,compression=compression_opts)

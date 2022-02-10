import csv

time = [[6,7,8,9,10],[0,2,3,4,5]]
f = open('test.csv','w')
for t in time:
    for i in range(len(t)):
        if i == 0:
            f.write(str(t[i]))
        else:
            f.write(','+str(t[i]))
    f.write('\n')
f.close()

    

# a= [[1,2,3,4],[5,6,7,8]]
# print(len(a))
# f = open('test.csv', 'w')
# for item in a:
#     for i in range(len(item)):
#         if i == 0:
#             f.write(str(item[i]))
#         else:
#             f.write(',' + str(item[i]))
#     f.write('\n')
# f.close()
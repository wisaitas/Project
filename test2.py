import pandas as pd
 
# # List of Tuples

# students = [[1,2,3,4],[5,6,7,8]]

# strstudents = []
# for i in range(2):
#     i = i+1
#     strstudents.append(['t={}'.format(i)])

# # Create a DataFrame object
# df = []
# df.append(pd.DataFrame(students[0],columns=strstudents[0]))
# df.append(pd.DataFrame(students[1],columns=strstudents[1]))
# # dfcon = [df[0],df[1]]
# df1 = pd.DataFrame(students[0],columns=strstudents[0])
# df2 = pd.DataFrame(students[1],columns=strstudents[1])
# df3 = pd.Series(students[0])
# df4 = pd.Series(students[1])
# df5 = df3.append(df4)
# print(df5)
# # con = pd.concat(dfcon,axis=1)
# # print(con)

df = pd.read_csv('signal_dataset.csv')
print(df)

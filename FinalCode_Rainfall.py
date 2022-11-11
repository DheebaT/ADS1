
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno


# Used this data from MET to show the rainfall in england over the years
rainfall_df = pd.read_excel("Rainfall_data.xlsx")

#checking whether there are nay missing values in a data set
rainfall_df.isnull()

#checking the number of missing values in each column
rainfall_df.isna().sum()

#counting number of missing values in the dataset
print(rainfall_df.isnull().values.sum())

#visualising missing values
msno.bar(rainfall_df)
plt.show()
plt.savefig('Missing_data.png')

#changing the data into dataframe
df = pd.DataFrame(rainfall_df)

#dropped the missing values
df= df.dropna()

#printing the cleaned data set
print(df)


plt.figure()

#Line plot

# Used year and seasons to visualise the rainfall in england as a line plot
plt.plot(df['year'], df['win'], label = 'Winter')
plt.plot(df['year'], df['spr'], label = 'Spring')
plt.plot(df['year'], df['sum'], label = 'Summer')
plt.plot(df['year'], df['aut'], label = 'Autumn')

#to show the title of the visualised data
plt.title('Rainfall in England')


plt.legend()

# to save the visualised data in a png format
plt.savefig('Rainfall_lineplot.png')

# to show the visualisation of given data
plt.show()


#Bar plot
# Used year and seasons to visualise the rainfall in england as a Bar plot
plt.bar(df['year'], df['win'], label = 'Winter')
plt.bar(df['year'], df['spr'], label = 'Spring')
plt.bar(df['year'], df['sum'], label = 'Summer')
plt.bar(df['year'], df['aut'], label = 'Autumn')

plt.title('Rainfall in England')
plt.legend()

plt.savefig('Rainfall_barplot.png')
plt.show()


#Pie chart
# Used seasons to visualise the rainfall in england as a Pie chart
df_pie = df.tail(5)
print(df_pie)

plt.figure(dpi=144)

#to create a subplot for 4 seasons
plt.subplot(2,2,1)

#showing the pie chart for winter with percentage for last five years 
plt.pie(df_pie['win'], labels = df_pie['year'], autopct='%1.0f%%')
plt.title('Winter')

plt.subplot(2,2,2)
plt.pie(df_pie['spr'], labels = df_pie['year'], autopct='%1.0f%%')
plt.title('Spring')

plt.subplot(2,2,3)
plt.pie(df_pie['sum'], labels = df_pie['year'], autopct='%1.0f%%')
plt.title('Summer')

plt.subplot(2,2,4)
plt.pie(df_pie['aut'], labels = df_pie['year'], autopct='%1.0f%%')
plt.title('Autumn')

plt.show()
plt.savefig('Rainfall_Pieplot.png')

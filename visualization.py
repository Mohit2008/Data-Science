# 1.) Create a matrix of scatter plot
import pandas as pd
pd.scatter_matrix(data_frame)
sns.pairplot(iris);

# 2.) To identify groups in your data
df.mean().sort_values().plot(style='.')

#3.) Scatter plot using seaborn
sns.relplot(x="total_bill", y="tip", hue="size", data=tips);

#4.) Scatter plot using seaborn
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
sns.jointplot(x="x", y="y", data=df);
sns.jointplot(x="x", y="y", data=df, kind="kde"); #kernel density estimation

#5.) Line plot using seaborn
sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            kind="line", data=fmri);

#6.) Facet using seaborn
sns.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips);

#7.) Categorical scatter plot using seaborn
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);

#8.) Box plot using seaborn
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
sns.catplot(x="color", y="price", kind="boxen",data=diamonds.sort_values("color"));
sns.catplot(x="fare", y="survived", row="class",
                kind="box", orient="h", height=1.5, aspect=4,
                data=titanic.query("fare > 0"))

#9.) Voilin plot using seaborn
sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", split=True, data=tips);

#10.) Bar plot using seaborn
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);

#11.) Count plot using seaborn
sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic);
sns.countplot(y="deck", data=titanic, color="c");

#12.) Distribution plot using seaborn
sns.distplot(x);

#13.) Linear relationship plot using seaborn
sns.regplot(x="total_bill", y="tip", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex",markers=["o", "x"], data=tips);

# Univariate analysis- Boxplot, histogram ,barchart
# Bivariate analysis:
#a)Quantitative vs Quantitative-scatter plot, regression line, correlation
#b)Quantitative vs Categorical-Box Plot#Bivariate Analysis,Bar Chart#Bivariate Analysis,Histogram#Bivariate Analysis, Density Plot#Bivariate Analysis (Can make use of Anova, z test or t test)
#c.)Categorical vs Categorical- Stacked Column Chart (Can use chi-square test)


#14)Time series decomposition plot
result = seasonal_decompose(series, model='multiplicative', freq=1)
result.plot()
pyplot.show()

#15 Time series visualisation 
import seaborn as sns
import matplotlib.pyplot as plt
sns.tsplot(data=df, time="date", unit="country",
           condition="Income Level", value="HIV Rate")
plt.show()

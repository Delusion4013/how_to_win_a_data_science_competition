# Exploratory data analysis (EDA)

> EDA is an art!

## What and why?

EDA allows to:

- Better understand the data
- Build an intuition about the data
- Generate hypothesizes
- Find insights (features useful for analysis)

## Things to explore

### Building Intuition about the data

1. Getting domain knowledge

   Prefer to have domain related knowledge. Help to understand the problem.

2. Checking if the data is intuitive

   E.g. age above 200? and agrees with domain knowledge?

3. Understanding how the data was generated

   > Understanding the generation process is crucial to set up a proper validation scheme

### Exploring anonymized data

> Anonymized data - direct data hide from original source for data privacy protection.

- Explore individual features

  - Guess the meaning of the columns

  - Guess the feature types

    `df.dtypes`: Guesses the type of the data frame (flawed, integer, object)

    `df.info()`

    `x.value_counts()`: count the unique values of the data

- Explore feature relations

  - Find relation between pairs
  - Find feature groups



## Exploration and visualization tools

Visualization -> idea : patterns lead to questions

Idea -> Visualization : Hypothesis testing

#### Explore individual features

- Histograms

  - `plt.hist(x)`

- Plots

  `plt.plot(x,'.')`

  `plt.scatter(range(len(x)), x, c=y)`

- Statistics

  `df.describe()` - get the basic information about features (mean,min, max,  std...)

  `x.mean()` `x.var()`

- Others

  `x.isnull()`

#### Explore feature relations

- Pairs:

  - Scatter plots

  - <img src="img/Week 2 - Exploratory data analysis/image-20210206203551184.png" alt="image-20210206203551184" style="zoom:80%;" />

    `plt.scatter(x1, x2)`

  - Correlation plots

    `df.corr(),  plt.matshow()`

- Groups

  - Corrplot  + clustering

  - plot (index vs feature statistics)

    `df.mean().sort_values().plot(style='.')`

#### Principles

1. Never make conclusion based on one single graph

## Dataset cleaning

### Constant features & Duplicated features

`trainset.nunique (axis=1) ==1`

`trainset.T.drop_duplicates()`

Get rid of same categorical feature using different indexes:

```python
for f in categorical_feats:
	traintest[f] = raintest[f].factorize()
	
trainset.T.drop_duplicates()
```

### Duplicated rows

- Check if same rows have same label
- Find duplicated rows, understand why they are duplicated.

### Check If the dataset is shuffled

Plot target value vs. row index.



## EDA check list

- Get domain knowledge

- Check if the data is intuitive

- Understand how the data was generated

  <hr>

- Explore individual features

- Explore pairs and groups

  <hr>

- Clean features up

  <hr>

- Check for leaks!

## Additional Material and  links

### Visualization tools

- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plot.ly/python/)
- [Bokeh](https://github.com/bokeh/bokeh)
- [ggplot](http://ggplot.yhathq.com/)
- [Graph visualization with NetworkX](https://networkx.github.io/)

### Others

- [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

## Kaggle competition EDA example


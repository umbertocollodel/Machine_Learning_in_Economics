# Machine_Learning_in_Economics
Repository with material for the course "Machine Learning in Economics" taught by Philipp Ketz and Hannah Bull (PSE 2020-2021)

## Homework 1

We aim to predict whether or not tweets are made by realDonaldTrump or BernieSanders. This is a useful task if, say, we want to study a certain population on twitter and we would like to know their political views, or if we want to create a left-right/Bernie-Trump index to use as a variable. The training dataset contains around 2500 tweets from the past few weeks in approximately equal proportions from Bernie and Trump. The test dataset contains 633 unlabelled tweets.

Estimated out-of-sample probabilities:

<img src="https://user-images.githubusercontent.com/33840988/166444875-a836228f-10d5-4a2f-93fb-1b95b507b7be.png" width="650" />

## Homework 2

Part 1: we simulate the treatment coefficient in presence of a large number of controls with different methods using generated data. We calculate the expected value of each estimator and discuss it.

Part 2: we create a custom function for the double ML method described in Chernozukov et al. (2018) with k-fold validation.

Part 3: we apply the double machine learning function on real data. The following part is based on the article “Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment” by Gerber, Green and Larimer (2008). The article is about a very large field study where registed voters in the US were randomly selected and received letters encouraging them to vote. As the electoral role in the US is public and one can consult who voted, the researchers were able to analyse the effect of the letters on voter turnout. The data contains variables relating to the treatment (X), response (Y) and socio-electoral characteristics (W). We want to use double machine learning to estimate the coefficient β of X


An extract:

```
k2ml <- function(X, W, Y, K, SL.library.X,  SL.library.Y, family.X, family.Y)

```



## Homework 3 


The dataset for this homework comes from a very large survey in the US, where participants were randomly asked one of two questions with similar wording:

"We are faced with many problems in this country, none of which can be solved easily or inexpensively. I’m going to name some of these problems, and for each one I’d like you to tell me whether you think we’re spending too much money on it, too little money, or about the right amount. Are we spending too much, too little, or about the right amount on (welfare/assitance to the poor)?"

The treatment W is the wording of the question (0/1). The outcome Y is that the person surveyed thinks the government spends too much.

Is there treatement effect heterogeneity?
Where is this heterogeneity?


<img src="https://user-images.githubusercontent.com/33840988/166445921-4c497aaf-b6f5-4f9f-ad05-1d470fbc2766.png" width="650" />





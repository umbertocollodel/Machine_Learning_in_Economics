# Machine_Learning_in_Economics
Repository with material for the course "Machine Learning in Economics" taught by Philipp Ketz and Hannah Bull (PSE 2020-2021)

## Homework 1

We aim to predict whether or not tweets are made by realDonaldTrump or BernieSanders. This is a useful task if, say, we want to study a certain population on twitter and we would like to know their political views, or if we want to create a left-right/Bernie-Trump index to use as a variable. The training dataset contains around 2500 tweets from the past few weeks in approximately equal proportions from Bernie and Trump. The test dataset contains 633 unlabelled tweets.

Results:

![Image](../Machine_learning_for_economics_material/output/homework_1/figures/fitted.png)

## Homework 2

Part 1: we simulate the treatment coefficient in presence of a large number of controls with different methods using generated data. We calculate the expected value of each estimator and discuss it.

Part 2: we create a custom function for the double ML method described in Chernozukov et al. (2018) with k-fold validation.

Part 3: we apply the double machine learning function on real data. The following part is based on the article “Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment” by Gerber, Green and Larimer (2008). The article is about a very large field study where registed voters in the US were randomly selected and received letters encouraging them to vote. As the electoral role in the US is public and one can consult who voted, the researchers were able to analyse the effect of the letters on voter turnout. The data contains variables relating to the treatment (X), response (Y) and socio-electoral characteristics (W). We want to use double machine learning to estimate the coefficient β of X


```
k2ml <- function(X, W, Y, K, SL.library.X,  SL.library.Y, family.X, family.Y)

```

library(tidyverse)
library (ISLR)
library(ggplot2)
library(rcompanion)
library(rpart)
library(rpart.plot)
library(formattable)
library(aod)
library(MASS)
library(caret)
library(data.tree)
library(caTools)
library(psych)
library(class)

install.packages("devtools")
library(devtools)
install_github("fawda123/ggord")
library(ggord)

#to load the dataset 
credit44 <- read.csv("~/Documents/Datasets/credit44.csv")

str(credit44)
head(credit44)
lapply(credit44, class)
credit44

#create a dataframe for the dataset for complete cases 
#create backup of the dataset in a dataframe
df.bkp <- credit44
df <- credit44[complete.cases(credit44),]

# Set up factors.
df$Creditability <- as.factor(df$Creditability)
df$Account.Balance <- as.factor(df$Account.Balance)
df$Payment.Status.of.Previous.Credit <- as.factor(df$Payment.Status.of.Previous.Credit)
df$Purpose <- as.factor(df$Purpose)
df$Value.Savings.Stocks <- as.factor(df$Value.Savings.Stocks)
df$Length.of.current.employment <- as.factor(df$Length.of.current.employment)
df$Sex...Marital.Status <- as.factor(df$Sex...Marital.Status)
df$Guarantors <- as.factor(df$Guarantors)
df$Most.valuable.available.asset <- as.factor(df$Most.valuable.available.asset)
df$Concurrent.Credits <- as.factor(df$Concurrent.Credits)
df$Type.of.apartment <- as.factor(df$Type.of.apartment)
df$Occupation <- as.factor(df$Occupation)
df$Telephone <- as.factor(df$Telephone)
df$Foreign.Worker <- as.factor(df$Foreign.Worker)

str(df)

#preliminary visualisation
creditability.plot <- ggplot(data = df, aes(x = Age..years., 
                                           y = Duration.of.Credit..month., 
                                           color = Creditability)) +
  geom_point()
creditability.plot

creditability.plot + scale_color_gradientn(
  colours = rainbow(5, start = 0, end = .8))

#to check the distribution 
plotNormalHistogram(df$Credit.Amount)
plotNormalHistogram(log(df$Credit.Amount))

#optionally, we could add the log of credit amount to the df dataframe
log.credit.amount <- log(df$Credit.Amount)
df <- data.frame(df, log.credit.amount)

#create the training and test dataframe for the decision tree
set.seed(777)
sample = sample.split(df$Creditability, SplitRatio = 0.7)
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)

#decision tree classification with training data
tree <- rpart(Creditability ~ ., 
              method = "class",
              data = train)

#predictions
tree.creditability.predicted <- predict(tree, test, type = "class")

#confusion matrix for evaluating the model
confusionMatrix(tree.creditability.predicted, test$Creditability)

#visualizing the decision tree
prp(tree)
rpart.plot(tree, type = 0,
           fallen.leaves = F,
           cex = .5)


###Logistics Regression
#splitting the data to train and test the model
set.seed(666)
split <- sample.split(df, SplitRatio = 0.8)
glmTrain <- subset(df, split == "TRUE")
glmTest <- subset(df, split == "FALSE")

#table of categorical outcome (i.e. Creditability) and variable (Account.Balance)
xtabs(~Creditability + Account.Balance, data = df)


#predicting the creditability using some independent variables in the dataset
#using two non-categorical variables and one categorical variables
logmodel <- glm(Creditability ~ Credit.Amount + 
                  Duration.of.Credit..month.
                + Account.Balance,
                data = glmTrain, family = "binomial")
summary(logmodel)

#odds ratio for each variable in the regression model
exp(coef(logmodel))

#odds ratio and 95% confidence interval
exp(cbind(OddsRatio = coef(logmodel), confint(logmodel)))

#confidence intervals using profiled log-likelihood
confint(logmodel)

#confidence intervals using standard errors
confint.default(logmodel) 

#Wald test for the overall effect of the Account Balance variable
wald.test(b = coef(logmodel), Sigma = vcov(logmodel), Terms = 4:6)

## Wald Test (for the different levels of Account Balance, e.g. 2 and 3)
level <- cbind(0, 0, 0, 1, -1, 0) # create a vector to contrast 
wald.test(b = coef(logmodel), Sigma = vcov(logmodel), L = level)

#Odds ratio
exp(coef(logmodel))

#run the test data through the model
resp <- predict(logmodel, glmTest, type = "response")
resp

resp <- predict(logmodel, glmTrain, type = "response")
resp

#validate the model using confusion matrix
confmatrix <- table(
  Actual_Value = glmTrain$Creditability, Predicted_Value = resp >0.5
)
confmatrix

#measuring accuracy
(confmatrix[[1,1]] + confmatrix[[2,2]]) / sum(confmatrix)

#Predicting the probabilities
#Calculating the probabilities of creditability at each account balance
df.n1 <- with(df, data.frame(Credit.Amount = mean(Credit.Amount), 
                             Duration.of.Credit..month. = mean(Duration.of.Credit..month.), 
                             Account.Balance = factor(1:4)))
df.n1
df.n1$Account.BalanceP <- predict(logmodel, newdata = df.n1, type = "response")
df.n1

df.n2 <- with(df, data.frame(Credit.Amount = rep(seq(from = 2000, to = 8000, length.out = 100),4), 
                             Duration.of.Credit..month. = mean(Duration.of.Credit..month.), 
                             Account.Balance = factor(rep(1:4, each = 100))))
view(df.n2)

# add log odds, probabilities and include standard error for confidence intervals
df.n3 <- cbind(df.n2, predict(logmodel, newdata = df.n2, type = "link", se = TRUE))
df.n3 <- within(df.n3, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})
view(df.n3)

# Plot showing predicted probabilities against credit amount ranging from 2000 to 8000
ggplot(df.n3, aes(x = Credit.Amount, y = PredictedProb)) + 
  geom_ribbon(aes(ymin = LL, ymax = UL, fill = Account.Balance), alpha = 0.2) + 
  geom_line(aes(colour = Account.Balance), size = 1)

#Likelihood ratio test
summary(logmodel)
logLik(logmodel)
with(logmodel, null.deviance - deviance) # find the difference in deviance
with(logmodel, df.null - df.residual) # The df for the difference between the two models = the number of predictor variables
with(logmodel, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)) # obtain p-value


#Discriminant analysis
#split the data into testing and training
set.seed(666)
ind <- sample(2, nrow(df),
              replace = TRUE,
              prob = c(0.8, 0.2))

ldaTrain <- df[ind==1,]
ldaTest <- df[ind==2,]

#plot pairs of all non-categorical variables for the DA
pairs.panels(
  df[,c("Duration.of.Credit..month.","Credit.Amount","Instalment.per.cent",
        "Duration.in.Current.address","Age..years.","No.of.Credits.at.this.Bank")],
  gap = 0, pch = 21
)

#do lda(Linear Discriminant Analysis) with equal priors using training data
credit.lda <- lda(
  Creditability ~ Duration.of.Credit..month. + Credit.Amount +
    Instalment.per.cent + Duration.in.Current.address + 
    Age..years. + No.of.Credits.at.this.Bank,
  data = ldaTrain, prior = c(1,1)/2)

credit.lda

credit.lda.predict <- predict(credit.lda, ldaTrain)

tab <- table(ldaTrain$Creditability, data = credit.lda.predict$class)
addmargins(tab)

#express table entries as fraction Of marginal table
round(addmargins(prop.table(tab,1)*100,2),2)

# prop.table(.tab,1) 
ncorrect <- sum(diag(tab))

# diag(tab) #
ntotal <- sum(tab)
cat(ncorrect," correctly allocated out of ",ntotal," (",100*ncorrect/ntotal,"%)","\n")


#confusion matrix and accuracy - training data
p1 <- predict(credit.lda, ldaTrain)$class
table(Predicted = p1, Actual = ldaTrain$Creditability)

#confusion matrix and accuracy - testing data
p2 <- predict(credit.lda, ldaTest)$class
tabtest <- table(Predicted = p2, Actual = ldaTest$Creditability)
sum(diag(tabtest))/sum(tabtest)


#look at misclassified cases in the training data
ldaTrain$predicted_credit <- credit.lda.predict$class
ldaTrain[ldaTrain$Creditability != ldaTrain$predicted_credit, ]


#K-nearest neighbors algorithm
#load df dataframe into a new dataframe for KNN
knn.df <- df

#load KNN dataframe with some variables to be used for KNN
knn.df <- knn.df[,c(
  "Creditability", "Duration.of.Credit..month.","Credit.Amount",
  "Instalment.per.cent","Duration.in.Current.address","Age..years.",
  "No.of.Credits.at.this.Bank")]

view(knn.df)

#function to normalize the data
norm <-function(x) { (x -min(x))/(max(x)-min(x)) }

#create new dataframe containing normalised data
knn.df_norm <- as.data.frame(lapply(knn.df[,-1],norm))
summary(knn.df_norm)

#split data into training and testing 
set.seed(666)
rand <- sample(1:nrow(knn.df_norm), size = nrow(knn.df_norm)* 0.8,
               replace = FALSE)

train_knn <- knn.df[rand,]
test_knn <- knn.df[-rand,]

# creating a dataframe for the target feature "creditability"
train_knn_target <- knn.df[rand, 1]
test_knn_target <- knn.df[-rand, 1]

#run KNN algorithm
pr <- knn(train = train_knn, test = test_knn, cl = train_knn_target, k = 29)

#create confusion matrix 
confusionMatrix(table(pr, test_knn_target))

pr_tab <- table(pr,test_knn_target)
pr_tab

#measure the accuracy of the model
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100} 
accuracy(pr_tab)

#to find the optimal k value
i = 1
k.optm = 1
for (i in 1:51){
  knn.mod <- knn(train = train_knn, test = test_knn, cl = train_knn_target, k = i)
  k.optm[i] <- 100 * sum(test_knn_target == knn.mod)/NROW(test_knn_target)
  k = i
  cat(k, '=', k.optm[i], '\n')
}

plot(k.optm, 
     main = "Plot showing the optimal value of K",
     type = "b", xlab = "K-Value", ylab = "Accuracy Level")



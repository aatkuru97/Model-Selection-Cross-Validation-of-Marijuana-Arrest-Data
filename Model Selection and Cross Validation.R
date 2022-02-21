#Looking at the "Arrests for Marijuana Possession" dataset using R
library(faraway)
library(carData) #Companion to Applied Regression Data Sets
library(epiDisplay)#to use the function logistic.display for an easier view of the logistic model
library(StepReg)#package for model selection of a logistic regression model 
head(Arrests) #first 6 observations
str(Arrests) #data information
summary(Arrests) #summary of the "Arrests for Marijuana Possession" dataset
#There are 5226 observations with 8 variables 
#Model Selection

data(Arrests, package="carData") #loads the specified data set
#We are trying to find a model that tells us the relationship of if someone who was arrested for marijuana possession was released with a summons and specific factors of the arrestee
#We use a logistic regression since our dependent variable is categorical (Yes or No). Below is our complete model:
logmodel <- glm(released~ colour+ year + age + sex + employed + citizen + checks, data = Arrests, family = "binomial")
summary(logmodel) #provide summary
logistic.display(logmodel) #easier view 
x <- model.matrix(logmodel) #optional put in dataframe
head(x) #summary of first 6
#We want to know if we need to include all independent variables of the dataset in this model.
#Fewer variables decreases chance of increasing variance of dependent variable, which will reduce chances of overfitting. 
#We will use model selection to find which independent variables are significant in predicting the dependent variable/
#Generate all possible and stepwise selection procedures on the 7 predictors provided in the logmod statement
#For Logistic Regression:
#Akaike information criterion: AIC = 2p + 2*abs.value(Log Likelihood)
# where p = number of parameters

# Also, Residual Deviance of a Model is equal to 2*abs(Log Likelihood of the Model)

#Check AIC computation for Full model  p=7
##AIC (Full Model) = Residual Deviance of Full Model + 2p   = 199.52 + 2*7 = 215.52
##Log Likelihood of Full Model = -(Residual Deviance/2)  =  -199.52/2 = - 99.76 

anova(logmodel, test="Chisq") #can be used to compare nested models
#By this point, we already have an idea that the most significant variables to y are checks, colour, employed, and citizen but we want to still want to see what models we will get if we do the different types of selection.
y<- "released"
stepwiselogit(data=Arrests,y, exclude = NULL, include = NULL, selection = "forward",
              select = "AIC", sle = 0.15, sls = 0.15)

stepwiselogit(data=Arrests,y, exclude = NULL, include = NULL, selection = "backward",
              select = "AIC", sle = 0.15, sls = 0.15)

stepwiselogit(data=Arrests,y, exclude = NULL, include = NULL, selection = "bidirection",
              select = "AIC", sle = 0.15, sls = 0.15)

#obtain the percentage change in odds (pi/(1-pi)) for every
# 1-unit increase in Xi, holding all other X's fixed
(exp(coef(logmodel)) - 1) * 100 #used to interpret odds ratio - finds coefficients of logmodel
##-----------------------##

#We found that no matter what model selection technique we did, there are the same 4 significant independent variables for fitting our dependent variable and the model is the same model for each. Note that the lower AIC, the better.
#But we need to know if there is still a better model where we can avoid overfitting. 
#Since we already know that no matter what selection technique we do, we get the same model, we can try to compare the model we got with a model that has the 3 most significant indpendent variables.
#Only looking at the AIC to determine which model you would choose is generally not enough in making sure you selected the best model.
#We need to assess the accuracy and validity of each model to determine which model we should choose by Cross Validation or CV.
#The technique we will use for CV is K-Fold Cross Validation because it has the advantage of using all data for estimating the model over other CV techniques.
#Especially since over 5,000 observations and is large, we could definitely do a 10-Fold Cross Validation.
#model chosen by forward selection
forward <- glm(released  ~ checks + employed + citizen + colour, data=Arrests, family = "binomial")
summary(forward) 
exp(coef(forward)) #find beta parameters of the 4 variable forward selection model
(exp(coef(forward)) - 1) * 100 #find the  percentage change in odds for every 1 unit increase in x
logistic.display(forward) #display the results of the forward model (optimal model selected through model selection) 
#looking at the forward selection and finding the three most signficant variables
forward2 <- glm(released ~ checks + employed + citizen, data=Arrests, family = "binomial")
summary(forward2)
forward3 <-glm(released~checks+employed, data=Arrests, family="binomial")
summary(forward3)

#Note that by each selection, there are the same predictors used in each model
##-------------------------------------------------------------------------------------##
#Start the cross-validation to choose between the model chosen by forward selection
# and the model chosen by backward elimination
# and the model choosen by bidirection selection 
require(caret) #package used for cross validation 
library(e1071)

# Define training control
set.seed(13245)
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model_forward <- train(released ~checks + employed + citizen + colour,data = Arrests, method = "glm",
                       trControl = train.control)
# Summarize the results
print(model_forward)
##

# Define training control
set.seed(14235)
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model_forward2 <- train(released ~ checks + employed + citizen,data = Arrests, method = "glm",
                        trControl = train.control)
# Summarize the results
print(model_forward2)
##
#We noticed the accuracy and reliability with the model with the three predictors is higher than the model we were given through model selection.
#Hence, we found a better model.
#To check and see if we kept the two most significant variables from forward selection, would we even get a better model?
# Define training control
set.seed(14235)
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model_forward3 <- train(released ~ checks + employed,data = Arrests, method = "glm",
                        trControl = train.control)
# Summarize the results
print(model_forward3)
#The answer is no. The accuracy and reliability go down again.
#So we can conclude from cross validation, that our best model for our response variable, released, is our model_forward2.
#Assessing the predictive ability of the model we choose, forward2
#Looking at the ROC Curve
library(pROC)
#generate ROC curve
ROCresult <- roc(Arrests$released ~ forward2$fitted)
plot(ROCresult, legacy.axes = TRUE)
names(ROCresult)
ROCresult$auc

#would like to see how the model is doing as a classifier
#Our decision boundary will be 0.5. If predicted probability of P(released|checks+employed+citizen) > 0.5 then predicted.released = 1 otherwise predicted.released=0
#Note that for some applications different thresholds than 0.5 could be a better option
#we should be using test data and should be performing Cross Validation
#we are using the training data set so overfitting is a concern
predicted.released <- predict(forward2,data=Arrests,type='response')#using the type='response' option generates P(released|at each level of the predictors) 
predicted.released <- ifelse(predicted.released >0.5, "Yes","No")  #predicted.released is 1 if predicted P(released|checks+employed+citizen) > 0.50, 0 otherwise
table(predicted.released, Arrests$released)
arrests <- cbind(Arrests,predicted.released)
data.frame(arrests)
predicted.released <- as.factor(predicted.released)  #need to be sure this variable is a factor
Arrests$released <- as.factor(Arrests$released)  #need to be sure this variable is a factor
confusionMatrix(Arrests$released,predicted.released)  #found in the caret library
##
##
####End of 10-fold Cross Validation####










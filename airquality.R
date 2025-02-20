#we load the dataset from R directory
datasets::airquality

#install different libraries 
install.packages("caret")
install.packages("ggthemes")
install.packages("lattice")
install.packages("e1071")
install.packages("nnet")
install.packages("tidyverse")
install.packages("car")
install.packages("class")
install.packages("naivebayes")

#libraries loaded
library(ggthemes)
library(ggplot2)
library(caret)
library(ggiraphExtra)
library(ggplot2)
library(broom)
library(readr)
library(MASS)
library(e1071)
library(nnet)
library(corrplot)
library(tidyverse)
library(car)
library(class)
library(naivebayes)
library(boot)

#we view our dataset
View(airquality)

#understanding data structure
str(airquality)

#we remove all the N/A values from the dataset
#we call the non NA dataset na
na<- na.omit(airquality)
print(na)

str(na)

#summarise cleaned dataset
summary(na)

#we will split the dataset into subset of 80:20(trainging:validation)
split_data <- createDataPartition(na$Month, p=0.8, list=FALSE)
testset <- na[-split_data,]
trainset <- na[split_data]

#Summarise the training dataset
summary(trainset)

#We plot histogram
ggplot(data = na,mapping =  aes(Ozone))+geom_histogram( bins = 10)
ggplot(data = na,mapping =  aes(Temp))+geom_histogram(bins = 10)
ggplot(data = na,mapping =  aes(Wind))+geom_histogram(bins = 10)
ggplot(data = na,mapping =  aes(Solar.R))+geom_histogram(bins = 10)

#We plot boxplpot
ggplot(data = na,mapping =  aes(-1,Ozone))+geom_boxplot()
ggplot(data = na,mapping =  aes(-1,Temp))+geom_boxplot()
ggplot(data = na,mapping =  aes(-1,Wind))+geom_boxplot()
ggplot(data = na,mapping =  aes(-1,Solar.R))+geom_boxplot()


#featurePlot(x=na[,1:4], y=na[,5], plot="box", scales=list(x=list(relation="free"), y=list(relation="free")), auto.key=list(columns=3))

#we find correlation between the variables 
correlations <- cor(na[,1:5])
corrplot(correlations, method = "circle")

#overall plot  
plot(na)

#we convert months variable to characters
na$Month <- as.character(na$Month)
na$Month[na$Month == "5"] <- "may"
na$Month[na$Month == "6"] <- "june"
na$Month[na$Month == "7"] <- "july"
na$Month[na$Month == "8"] <- "august"
na$Month[na$Month == "9"] <- "september"
na

cor(na[,1:4])

#we will be building some plots using multinomial logistic Regression,
#linear Discriminant Analysis and K-nearest Neighbor
#Logistic regression
#We use logistic regression with two predictor
#1.first predictors are Wind and Temp according to month 
log_fit1=multinom(Month~Temp+Wind, data=na)
print(log_fit1)

#2.Second predictor are Ozone and Solar radiation according to month
log_fit2=multinom(Month~Ozone+Solar.R, data = na)
print(log_fit2)

#3.Next we use model logit with all predictors
log_fit_all=multinom(Month~Ozone+Solar.R+Wind+Temp, data = na)
print(log_fit_all)

AIC(log_fit_all)


#linear Discriminant analysis
#Model LDA1 with one predictor
mdl.1da1<-lda(Month ~ Ozone, data = na )
mdl.1da1
mdl.lda1.p<-predict(mdl.1da1, newdata = na[,c(1,2)])$class
mdl.lda1.p

#model LDA2 with two predictor
mdl.1da2<-lda(Month ~ Ozone + Solar.R, data = na )
mdl.1da2
mdl.lda2.p<-predict(mdl.1da2, newdata = na[,c(1,2)])$class
mdl.lda2.p

mdl.1da3<-lda(Month ~ Ozone + Wind, data = na )
mdl.1da3
mdl.lda3.p<-predict(mdl.1da3, newdata = na[,c(1,3)])$class
mdl.lda3.p

mdl.1da4<-lda(Month ~ Ozone + Temp, data = na )
mdl.1da4
mdl.lda4.p<-predict(mdl.1da4, newdata = na[,c(1,4)])$class
mdl.lda4.p

mdl.1da5<-lda(Month ~ Ozone + Solar.R + Wind + Temp, data = na )
mdl.1da5
mdl.lda5.p<-predict(mdl.1da5, newdata = na[,c(1:4)])$class
mdl.lda5.p

summary(mdl.lda5)

plot(mdl.1da3)
plot(mdl.1da2)


#determine how well the model fits
table.lda1 <- table(mdl.lda1.p,na[,5])
table.lda1

plot(table.lda1)

table.lda5 <- table(mdl.lda5.p,na[,5])
table.lda5

plot(table.lda1)



#k fold cross validation
#lda
K <- 10
fold <- cut(seq(1, nrow(na)), breaks = K, labels = FALSE)
head(fold)
set.seed(1)
cv.lda<-sapply(1:K, FUN = function(i){
  testID<- which(fold == i, arr.ind = TRUE)
  test <- na[testID, ]
  train<- na[-testID, ]
  ldaf<- lda(Ozone ~ Temp + Solar.R + Wind + Temp, data = na)
  lda.pred <- predict(ldaf, test)
  cv.est.lda <- mean(lda.pred$class !=test$Month)
  return(cv.est.lda)
}
)
cv.lda
mean(cv.lda)

BIC(fit)

#kfold With linear mean regression method

set.seed(1)
train.control <- trainControl(method = "cv", number = 10)
# Train the model1
modelkfold1 <- train(Ozone ~ Solar.R , data = na , method = "lm",
               trControl = train.control)
# Summarize the results of model 1
print(modelkfold1)

modelkfold1= lm(Ozone~ Solar.R, data = na)
AIC(modelkfold1)
BIC(modelkfold1)

# Train the model2
modelkfold2 <- train(Ozone ~  Solar.R + Temp, data = na , method = "lm",
                     trControl = train.control)
# Summarize the results of mode2 1
print(modelkfold2)

modelkfold2= lm(Ozone~ Solar.R + Temp, data = na)
AIC(modelkfold2)
BIC(modelkfold2)

# Train the model3
modelkfold3 <- train(Ozone ~ Solar.R + Temp + Wind , data = na , method = "lm",
                     trControl = train.control)
# Summarize the results of model 3
print(modelkfold3)

modelkfold3= lm(Ozone~ Solar.R + Temp + Wind, data = na)
AIC(modelkfold3)
BIC(modelkfold3)


3#knn\
library(caret)
trControl <- trainControl(method  = "cv",
                          number  = 5)
fit <- train(Ozone ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trControl,
             data       = na)
head(fit)
fit
fit= knn(Ozone~., data = na)
AIC(fit)
BIC(fit)


#LOOCV 
glm.fit<- glm(Ozone~ Temp, data = na)
summary(glm.fit)
cv.error <- cv.glm(na, glm.fit)
names(cv.error)
round(cv.error$delta, 2)
cv.error = rep(0,5)

for (i in 1:5) {
  glm.fit = glm(Ozone ~ poly(Temp,i), data=na)
  cv.error[i] = round(cv.glm(na, glm.fit)$delta[1],2)
}

cv.error

glm.fit<- glm(Ozone~ Temp, data = na)
summary(glm.fit)
cv.error <- cv.glm(na, glm.fit)
names(cv.error)
round(cv.error$delta, 2)
cv.error = rep(0,5)

for (i in 1:5) {
  glm.fit = glm(Ozone ~ poly(Temp,i), data=na)
  cv.error[i] = round(cv.glm(na, glm.fit)$delta[1],2)
}

cv.error

#


#data2 <- lda(Month ~ Ozone + Solar.R, data = na, CV = TRUE)#table2 <- table(data2$class, na[,5])
#table2
#plot(table2)

#fit the model
set.seed(7)
fit.knn <- train(Method~Ozone, data=na, method="knn", metric=metric, trControl=control)
print(fit.knn)

Modelm <- lm(Month~Ozone+ Solar.R+ Temp+ Wind, data = na)
print(Modelm)

#anova
Anova(Modelm)
#Results

dev.off()

qplot(Month, Ozone, data = na, geom = "boxplot", color = Month)


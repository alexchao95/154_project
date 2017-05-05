#Attributes data
attrib <- read.csv("attributes.train.csv")
#Complete Attributes data
attrib.complete <- read.csv("attributes.train.csv")

#Small Data Cleaning
attrib <- attrib[-1,c(-1,-2)]
attrib.complete <-attrib.complete[-1, c(-1,-2)]
attrib[is.na(attrib)] <- 0.5

#Split into train and test
my.rand.perm.attrib<-sample.int(n=2476)
size.training.data.attrib<-2476*0.75
training.data.attrib<-attrib[my.rand.perm.attrib[1:size.training.data.attrib],]
test.data.attrib<-attrib[my.rand.perm.attrib[(1+size.training.data.attrib):2476],]

dim(training.data.attrib)
dim(test.data.attrib)

#SVM on Attributes data
svm.model.attrib <- svm(stars~.,data=training.data.attrib, cost = 100, gamma = 1)
svm.pred.attrib  <- predict(svm.model.attrib, newdata=test.data.attrib[,-1])
actual.star<-test.data.attrib[,1]
agreement.Vector.attrib<-(abs((svm.pred.attrib*2)/2 - actual.star) < 0.5)
length.Test.Vector.attrib<-length(actual.star)
misClassif.attrib<-1-sum(agreement.Vector.attrib)/length.Test.Vector.attrib
misClassif.attrib
table(round(svm.pred.attrib*2)/2,actual.star)

MSE.svm <- rmse(svm.pred.attrib,actual.star)

#Kernel SVM
install.packages("kernlab")
library(kernlab)
ksvm.model.attrib<-ksvm(as.matrix(training.data.attrib[,2:35]), 
                        as.factor(training.data.attrib[,1]), 
                        type="C-svc",kernel='rbfdot',C=100000)
dim(training.data.attrib)

predictors.test.attrib<-as.matrix(test.data.attrib[,-1])
ksvm.pred.attrib<-predict(ksvm.model.attrib,predictors.test.attrib)
table(actual.star,ksvm.pred.attrib)
actual.star<-test.data.attrib[,1]
agreement.Vector.attrib.ksvm<-(abs((as.integer(ksvm.pred.attrib)*2)/2 - actual.star) < 0.5)
length.Test.Vector.attrib.ksvm<-length(actual.star)
misClassif.attrib<-1-sum(agreement.Vector.attrib.ksvm)/length.Test.Vector.attrib
misClassif.attrib
pred.int <- ksvm.pred.attrib
pred.int <- (as.integer(x) + 1)/2

MSE.ksvm <- rmse(pred.int, actual.star)
MSE.ksvm

#SVM on Complete data
attrib.complete <- attrib.complete[complete.cases(attrib.complete),]
dim(attrib.complete)
my.rand.perm.attrib.c<-sample.int(n=471)
size.training.data.attrib.c<-471*.75
training.data.attrib.c<-attrib.complete[my.rand.perm.attrib.c[1:size.training.data.attrib.c],]
test.data.attrib.c<-attrib.complete[my.rand.perm.attrib.c[(1+size.training.data.attrib.c):471],]

svm.model.attrib.c <- svm(stars~.,data=training.data.attrib.c, cost = 100, gamma = 1)
svm.pred.attrib.c  <- predict(svm.model.attrib.c, newdata=test.data.attrib.c[,-1])
actual.star.c<-test.data.attrib.c[,1]
agreement.Vector.attrib.c<-(abs((svm.pred.attrib.c*2)/2 - actual.star.c) < 0.5)
length.Test.Vector.attrib.c<-length(actual.star.c)
misClassif.attrib.c<-1-sum(agreement.Vector.attrib.c)/length.Test.Vector.attrib.c
misClassif.attrib.c
table(round(svm.pred.attrib.c*2)/2,actual.star.c)

RMSE.svm.c <- rmse(svm.pred.attrib.c - actual.star)
RMSE.svm.c

#Short linear regression
attrib.lm <- lm(attrib$stars~., data = attrib)
summary(attrib.lm)

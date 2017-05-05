library(tm)
library(nnet)
library(e1071)
library(randomForest)
library(bigmemory)


### read in data ###
review_train = read.csv('yelp_academic_dataset_review_train.csv')
review_test  = read.csv('yelp_academic_dataset_review_test.csv')
business_test<-read.csv("yelp_academic_dataset_business_test.csv")
####################

#TRAIN DATA PREP

### converting text of review into sparse dtm ###
reviews = as.vector(review_train$text)
reviews <- VCorpus(VectorSource(reviews))
reviews <- tm_map(reviews, content_transformer(tolower))
reviews <- tm_map(reviews, removePunctuation)
reviews <- tm_map(reviews, removeNumbers)
extra_stopwords<-c("pizza","food","coffee","tacos",
                   "mexican", "chicken","starbucks","restaurant","chinese",
                   "salsa", "burrito", "lunch", "taco", "place", "order",
                   "ordered", "ive", "also", "cheese", "eat", "sauce",
                   "rice", "people", "chips", "vegas", "hot", "menu", "drinks",
                   "beef", "drink", "asada", "carne", "fried", "meal", "salad",
                   "margaritas", "slice", "youre", "soup", "bar", "meal"
                   , "meat", "shrimp", "beans", "crust", "pork", "spicy",
                   "dinner", "breakfast", "portions", "tea", "burritos",
                   "guacamole", "egg", "chicago","onions","italian", "shredded",
                   "espresso", "youll", "scottsdale", "quesadilla", "chipotle",
                   "sausage", "mein", "chow", "corn", "rolls", "noodles",
                   "chili", "ice", "carnitas", "tortilla", "enchilada",
                   "water", "nachos", "iced" ,"plate", "garlic", "latte",
                   "pepperoni", "cream", "ate", "steak", "bean", "pizzas",
                   "tortillas", "margarita", "phoenix", "restaurants", "dish",
                   "enchiladas")
reviews <- tm_map(reviews, removeWords, c(stopwords("english"),extra_stopwords))
reviews <- tm_map(reviews, stripWhitespace)
dtm <- DocumentTermMatrix(reviews, control = list(weighting = weightTfIdf))
dtm = removeSparseTerms(dtm, 0.99)
##################################################


### convert dtm to big data frame ###
M=as.big.matrix(x=as.matrix(dtm))
dtm_df = as.data.frame(as.matrix(M))
#####################################


### upload positive and negative words and count them ###
neg_words = read.table("Desktop/negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("Desktop/positive-words.txt", header = F, stringsAsFactors = F)[, 1]

reviews$neg = sapply(reviews, tm_term_score, neg_words)
reviews$pos = sapply(reviews, tm_term_score, pos_words)
##########################################################


### bind columns for pos neg and stars and add . to colnames to avoid confusion ###
dtm_df$neg = reviews$neg
dtm_df$pos = reviews$pos
dtm_df$stars_ = as.factor(review_train$stars)

for (i in 1:ncol(dtm_df)) {
  names(dtm_df)[i] <- paste(names(dtm_df)[i], ".", sep="")
}
data<-dtm_df
####################################################################################

### find indices of 22 words that have highest tf-idf ###
avetfidf<-NULL
for (i in 1:ncol(data)) {
  avetfidf[i]<-mean(data[,i])
}
names(data)
indices<-order(avetfidf, decreasing=TRUE)[1:24]
names(data[,indices])
#########################################################

### split into train/test if not kaggle submission for CV ###
train<- sample(nrow(data),nrow(data)*.80)
data_train<- data[train,c(indices,725)]
data_test<- data[-train,c(indices,725)]
#############################################################

#----------------------------------------------------------------------------------

#TEST DATA PREP

### converting text of review into sparse dtm for kaggle test data ###
reviews_t = as.vector(review_test$text)
reviews_t <- VCorpus(VectorSource(reviews_t))
reviews_t <- tm_map(reviews_t, content_transformer(tolower))
reviews_t <- tm_map(reviews_t, removePunctuation)
reviews_t <- tm_map(reviews_t, removeNumbers)
extra_stopwords<-c("pizza","food","coffee","tacos",
                   "mexican", "chicken","starbucks","restaurant","chinese",
                   "salsa", "burrito", "lunch", "taco", "place", "order",
                   "ordered", "ive", "also", "cheese", "eat", "sauce",
                   "rice", "people", "chips", "vegas", "hot", "menu", "drinks",
                   "beef", "drink", "asada", "carne", "fried", "meal", "salad",
                   "margaritas", "slice", "youre", "soup", "bar", "meal"
                   , "meat", "shrimp", "beans", "crust", "pork", "spicy",
                   "dinner", "breakfast", "portions", "tea", "burritos",
                   "guacamole", "egg", "chicago","onions","italian", "shredded",
                   "espresso", "youll", "scottsdale", "quesadilla", "chipotle",
                   "sausage", "mein", "chow", "corn", "rolls", "noodles",
                   "chili", "ice", "carnitas", "tortilla", "enchilada",
                   "water", "nachos", "iced" ,"plate", "garlic", "latte",
                   "pepperoni", "cream", "ate", "steak", "bean", "pizzas",
                   "tortillas", "margarita", "phoenix", "restaurants", "dish",
                   "enchiladas")
reviews_t <- tm_map(reviews_t, removeWords, c(stopwords("english"),extra_stopwords))
reviews_t <- tm_map(reviews_t, stripWhitespace)
dtm <- DocumentTermMatrix(reviews_t, control = list(weighting = weightTfIdf))
##################################################


### convert dtm to big data frame ###
M=as.big.matrix(x=as.matrix(dtm))
dtm_df_test = as.data.frame(as.matrix(M))
#####################################

### adding columns for neg and pos ###
dtm_df_test$neg = sapply(reviews_t, tm_term_score, neg_words)
dtm_df_test$pos = sapply(reviews_t, tm_term_score, pos_words)
##########################################################


### add . to colnames to avoid confusion ###
for (i in 1:ncol(dtm_df_test)) {
  names(dtm_df_test)[i] <- paste(names(dtm_df_test)[i], ".", sep="")
}
##############################################

### keep columns for test that are same as those in train ###
data_test_kaggle <- dtm_df_test[, c(names(dtm_df_test) %in% names(dtm_df))]
#############################################################


#----------------------------------------------------------------------------------

#MISCLASSIFICATION ERROR/CROSS VALIDATION

### random forest prediction (non-kaggle) ###
ranfor<-randomForest(data_train[,-25], data_train[,25],
                      data=data_train, ntree=1000)
myRanforPredictions<-predict(ranfor, newdata=data_test)
rmse<-sqrt(mean((data_test$stars_.-myRanforPredictions)^2))
##############################################

### svm prediction (non-kaggle) ###
svm<-svm(stars_.~., data=data_train)
mySvmPredictions<-predict(svm, newdata=data_test)
rmse<-sqrt(mean((data_test$stars_.-mySvmPredictions)^2))
###################################

#----------------------------------------------------------------------------------

#KAGGLE PREDICIONS

### predictions and csv output for kaggle, random forest ###
ranfor<-randomForest(data[,-725],
                     y=data[,725],
                     data=data,
                     ntrees=1000)
myRandomForestPredictions<-predict(ranfor, newdata=data_test_kaggle)
data_test_kaggle$stars<-myRandomForestPredictions
business_prediction<-aggregate(as.numeric(data_test_kaggle$stars),
                               by=list(review_test$business_id),
                               FUN=mean, na.rm=TRUE)
names(business_prediction)=c("business_id", "stars")
business_prediction$stars<-round(business_prediction$stars*2)/2
remove_ind<-which(!(business_prediction$business_id %in% business_test$business_id))
business_prediction<-business_prediction[-remove_ind,]
write.table(business_prediction, file = "predictions.csv",
             sep=",", row.names = F)
##############################################

### predictions and csv output for kaggle, svm ###
ranfor<-randomForest(data[,-725],
                     y=data[,725],
                     data=data,
                     ntrees=1000)
myRandomForestPredictions<-predict(ranfor, newdata=data_test_kaggle)
data_test_kaggle$stars<-myRandomForestPredictions
business_prediction<-aggregate(as.numeric(data_test_kaggle$stars),
                               by=list(review_test$business_id),
                               FUN=mean, na.rm=TRUE)
names(business_prediction)=c("business_id", "stars")
business_prediction$stars<-round(business_prediction$stars*2)/2
remove_ind<-which(!(business_prediction$business_id %in% business_test$business_id))
business_prediction<-business_prediction[-remove_ind,]
write.table(business_prediction, file = "predictions.csv",
            sep=",", row.names = F)
##############################################


setwd("/Users/joshuamks/Google\ Drive/SchoolWork/Spring\ 2017/STAT\ 154/Final\ Project/Data")
library(quanteda)
library(quantreg)
library(quantmod)
library(ggplot2)
library(FNN)
library(kknn)
library(kernlab)
library(fda.usc)
library(glmnet)
library(pls)
library(MASS)
library(car)
library(DAAG)
library(tm)
library(nnet)
library(e1071)
library(bigmemory)
library(neuralnet)
library(Rcpp)



##Clustering, Boosting, Text Mining
business = read.csv("yelp_academic_dataset_business_train.csv")
business.test = read.csv("yelp_academic_dataset_business_test.csv")
checkin = read.csv("yelp_academic_dataset_checkin.csv")
review = read.csv("yelp_academic_dataset_review_train.csv")
review.test = read.csv("yelp_academic_dataset_review_test.csv")
tip = read.csv("yelp_academic_dataset_tip.csv")
users = read.csv("yelp_academic_dataset_user.csv")


##RAJ cleaning code
reviews = as.vector(review.test$text)
stopWords = c(stopwords("en"), "") 

cleanReview = function(review, stop_words=stopWords){
  # In your project, you could modify this function 
  # to modify a review however you'd like (e.g. 
  # add more stop words or spell checker -
  # This is a VERY preliminary version 
  # of this function)
  
  # Lowercase review 
  lower_txt = tolower(review)
  # Remove punctuation - (might want to keep !)
  lower_txt = gsub("[[:punct:]]", " ", lower_txt)
  # Tokenize review 
  tokens = strsplit(lower_txt, ' ')[[1]]
  # Remove stop words 
  clean_tokens = tokens[!(tokens %in% stopWords)]
  clean_review = paste(clean_tokens, collapse=' ')
  return(clean_review)
}

cleanCorpus = function(corpus){
  # You can also use this function instead of the first. 
  # Here you clean all the reviews at once using the 
  # 'tm' package. Again, a lot more you can add to this function...
  
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

clean_reviews = sapply(reviews, function(review) cleanReview(review))
review_corpus = Corpus(VectorSource(clean_reviews))

# If you use the 'cleanCorpus' function you would do 
review_corpus = cleanCorpus(Corpus(VectorSource(reviews)))

# Check out how first review looks like 
inspect(review_corpus[1])

# Create document term matrix - try TD-IDF
review_dtm = DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf)) # 93k terms

# Remove less frequent words
review_dtm = removeSparseTerms(review_dtm, 0.99) # 813 terms
inspect(review_dtm)

M=as.big.matrix(x=as.matrix(review_dtm))
review.counts = as.data.frame(as.matrix(M))

mean_tfidf<-NULL
for (i in 1:ncol(review.counts)){
  mean_tfidf[i]<-mean(review.counts[,i])
}
indices<-order(mean_tfidf, decreasing=TRUE)[1:500]
names(review.counts)[indices]
review.counts = review.counts[,c(indices)]
#############################

#write.csv2(review.text.df, file = "text 10thou", sep = ",", eol = "\n")

review.predictors = as.data.frame(cbind(review[,c("funny", "useful", "cool")], review.counts))
review.all = as.data.frame(cbind(review[,c("user_id", "business_id", "stars")], review.predictors))
#test dataset
review.test.counts = as.data.frame(cbind(review.test[,c("funny", "useful", "cool")], review.counts))
preserve = names(review.test.counts)[(names(review.test.counts) %in% names(review.predictors))]
review.test.counts = review.test.counts[, c(names(review.test.counts) %in% names(review.predictors))]

#lets try pc regression
review.predictors = review.predictors[,c(preserve)]
review.pcr.data = as.data.frame(cbind(review$stars, review.predictors))
review.pcr.data = review.pcr.data[,!duplicated(colnames(review.pcr.data))]
colnames(review.pcr.data)[1] <- "stars"

#lessthan = sample(1:nrow(review), 1000) #sampling a subset of the data

#review.pca = prcomp(review.pcr.data[lessthan,], center = TRUE, retx=TRUE)
#summary(review.pca)
#plot(review.pca$sdev^2,main="scree Plot, Covariance case",type="b")

#review.pcr.model = pcr(stars ~ ., data = review.pcr.data[lessthan,], validation = "CV", ncomp = 50)
#summary(review.pcr.model)
#review.pcr.model.fitted = review.pcr.model$fitted.values
#fitted = review.pcr.model.fitted[1:length(lessthan),1,20]

#review.pcr.business.aggr = cbind(fitted, review.all[lessthan,])
#colnames(review.pcr.business.aggr)[1] <- "fitted"
#review.pcr.business.aggr$mean.stars = ave(review.pcr.business.aggr$fitted, review.pcr.business.aggr$business_id)

#lookup = as.data.frame(unique(cbind(review.pcr.business.aggr$business_id, review.pcr.business.aggr$mean.stars)))
#colnames(lookup)[1] <- "id"
#colnames(lookup)[2] <- "mean.stars"
#colnames(business)[3] <- "id"

#library(plyr)
#business$id = as.numeric(business$id)
#plyr.pcr <- join(business, lookup, by = "id")
#plyr.pcr = plyr.pcr[!is.na(plyr.pcr$mean.stars),]
#plyr.pcr$mean.stars = round(plyr.pcr$mean.stars*2)/2
#sqrt(sum((plyr.pcr$mean.stars - plyr.pcr$stars)^2)/nrow(lookup))


#clustering
#lessthan = sample(1:nrow(review), 1000)
#numCenters = 50
#kmeans.review <- kmeans(review.predictors[lessthan,], numCenters, iter.max = 200, nstart = 1)
#kmeans.review$size
#test.review = as.data.frame(cbind(kmeans.review$cluster, review.all[lessthan,]))

#test.review$mean.stars = ave(test.review$stars, test.review$`kmeans.review$cluster`)
#test.review$mean.stars = round(test.review$mean.stars)
#sqrt(sum((test.review$mean.stars - test.review$stars)^2)/1000)
#takes an unnecessarily long ass time to run

##################################### neural net ##############################
maxs = apply(review.pcr.data, 2, max)
mins = apply(review.pcr.data, 2, min)
review.nn.0 = as.data.frame(scale(review.pcr.data, center = mins, scale = maxs))
review.nn = review.nn.0
colMeans(review.nn)

for (i in 1:ncol(review.nn)){
  colnames(review.nn)[i] = paste0(colnames(review.nn)[i],".x")
}
n = names(review.nn)
f <- as.formula(paste("stars.x ~", paste(n[!n %in% "stars.x"], collapse = " + ")))

lessthan = sample(1:nrow(review), 50000) #sampling a subset of the data
review.nn.train = review.nn[lessthan,]
review.nn.test = review.nn[-lessthan,]

nn.model = neuralnet(f, data=review.nn.train, hidden=c(50,5),
                     linear.output=T, threshold = 0.05, stepmax = 10000, lifesign = "full")
#review.nn.test.x = review.test.counts[,c(2:ncol(review.test.counts))]
review.nn.test.x = review.nn.test[,c(3:ncol(review.nn.test))]

pr.nn = compute(nn.model, review.nn.test.x)
pr.nn_ <- pr.nn$net.result*max(review.pcr.data$stars) + min(review.pcr.data$stars)
test.r <- review.nn.test$stars.x*max(review.pcr.data$stars) + min(review.pcr.data$stars)

MSE.nn <- sqrt(sum((test.r - pr.nn_)^2)/nrow(review.nn.test))

print("averaging and testing out the model on review.test.counts dataset and then averaging predictions per business, output as vector")
#review.test.counts = as.data.frame(cbind(review.test[,c("user_id","business_id")], review.test.counts))


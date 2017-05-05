#Alexander Chao
#05/05/17
#STAT 154
#SID 24427086

#####Processing Attributes dataset
#directories
setwd("~/Documents/YELP")
getwd()

#packages
library(plyr)
library(reshape2)
library(stringr)
library(glmnet)
library(lattice)


#Data
#Loading training data for business attributes
biz_train <- read.csv("yelp_academic_dataset_business_train.csv", stringsAsFactors = FALSE, na.strings = "")

#preliminary data exploration
str(biz_train)
str(biz_train$attributes)

####Processing business attributes variable
#separating business attribute variables
biz_train$attributes <- strsplit(biz_train$attributes, split=", ")
class(biz_train$attributes)


#determining the business with the maximum number of attributes
att <- biz_train$attributes
max_length <- length(att[[which.max(lapply(att, length))]])

#populating a matrix with the business attribute data in long format
x <- matrix(nrow= max_length*length(att), ncol = 3)
y <- lapply(att, function(x) str_split(x, pattern = ": "))

#note: titles of varialbe categories, ie 'Ambience' are removed here
for(j in 1:length(y)) {
  for (i in 1:length(y[[j]])) {
    if(length(y[[j]][[i]]) == 3) {
      y[[j]][[i]] <- y[[j]][[i]][2:3]
    }
    x[(i + ((j-1)*(max_length))),1:2] <- y[[j]][[i]]
    x[(i + ((j-1)*(max_length))),3] <- as.integer(j)
  }
}

#recasting business attribute data, removing extra punctuation
att_precast <- as.data.frame(x)
att_precast <- na.omit(att_precast)
att_precast$V1 <- str_replace_all(att_precast$V1, "[[:punct:]]", "")
att_precast$V2 <- str_replace_all(att_precast$V2, "[[:punct:]]", "")
att_precast$V3 <- as.integer(att_precast$V3)

att_precast$V1 <- as.factor(att_precast$V1)
att_precast$V2 <- as.factor(att_precast$V2)

att_cast <- dcast(att_precast,V3 ~ V1, value.var = "V2")

#Adding back business IDs and avg star ratings
for (i in 1:length(att_cast$V3)) {
  att_cast$business_id[i] <- biz_train$business_id[att_cast$V3[i]]
  att_cast$stars[i] <- biz_train$stars[att_cast$V3[i]]
}

att_cast <- att_cast[,c(54, 55, 2:53)]

#Counting number of NAs in business attribute data
count_nas <- sapply(att_cast, function (x) sum(is.na(x)))
plot(count_nas)
mean(count_nas)
which(count_nas > 300)

#Removing variables with more than 1500 NAs
rem <- which(count_nas > 1500)
att_cast <- att_cast[, -rem]


plot(count_nas)
mean(count_nas)

str(att_cast)

#removing discreet (non-binary variables)
att_cast <- att_cast[,!names(att_cast) %in% c("NoiseLevel", "RestaurantsAttire", "RestaurantsPriceRange2")]

#Converting Alcohol/Wifi to T/F
#test <- as.logical(att_df$BikeParking)
att_cast$Alcohol <- ifelse(att_cast$Alcohol == "beerandwine" | att_cast$Alcohol == "fullbar", "True", att_cast$Alcohol)
att_cast$Alcohol <- ifelse(att_cast$Alcohol == "none", "False", att_cast$Alcohol)
att_cast$WiFi <- ifelse(att_cast$WiFi == "free" | att_cast$WiFi == "paid", "True", att_cast$WiFi)
att_cast$WiFi <- ifelse(att_cast$WiFi == "no", "False", att_cast$WiFi)

#Converting all variables to logical
att_df <- NULL
att_ls <- NULL
#att_fact <- lapply(att_cast, as.factor)
#att_df <- as.data.frame(att_fact, stringsAsFactors = TRUE)

att_ls <- lapply(att_cast[3:36], as.logical)
att_df <- as.data.frame(att_ls, stringsAsFactors = TRUE)
att_df <- cbind(att_cast[,1:2], att_df)

#Adding in variable 'NAs' that counts number of missing attributes for each business
att_df$NAs <- rowSums(is.na(att_df))
summary(att_df$NAs)
cor(att_df[,c(2,37)])

#Processing 'categories' variable
summary(as.factor(biz_train$categories))
cat <- biz_train$categories

cat <- gsub(".*Chinese.*", "Chinese", cat)
cat <- gsub(".*Coffee.*", "Coffee", cat)
cat <- gsub(".*Mexican.*", "Mexican", cat)
cat <- gsub(".*Pizza.*", "Pizza", cat)
summary(as.factor(cat))

biz_train$categories <- as.factor(cat)

#Adding category and other attributes back to attributes dataset
att_df$business_id <- as.factor(att_df$business_id)
biz_train$business_id <- as.factor(biz_train$business_id)
att_df <- merge(att_df, subset(biz_train, select = c(business_id, categories, review_count, state)), by = 'business_id')
att_df$state <- as.factor(att_df$state)
att_df$categories <- as.factor(att_df$categories)


#writing the processed attribute data to csv
write.csv(att_df, file = "attributes.csv")


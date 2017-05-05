#Alexander Chao
#05/05/17
#SID 24427086
#STAT 154

#####Attributes data analysis
#directories
setwd("~/Documents/YELP")
getwd()

#packages
library(plyr)
library(reshape2)
library(stringr)
library(glmnet)
library(lattice)

#Reading in attribute data
att_df <- read.csv("attributes.csv", stringsAsFactors = TRUE)
str(att_df)
att_df <- att_df[,2:41]


#analyzing attributes by restaurant category
boxplot(review_count ~ stars, data = att_df, main = "number of reviews by average star rating", xlab = "stars", ylab = "num_reviews")
boxplot(stars ~ categories, data = att_df, main = "star rating by restaurant category", xlab = "category", ylab = "stars")
boxplot(stars ~ state, data = att_df, main = "star rating by state", xlab = "state", ylab = "stars")


#learning to cope with the missing values
#removes some variables and some observations such that there are no NAs
testing <- rowSums(is.na(att_df))
plot(testing)
rem_miss <- which(testing < 5)
test1 <- att_df[rem_miss,]

count_nas2 <- sapply(test1, function (x) sum(is.na(x)))
plot(count_nas2)
keep_var <- which(count_nas2 == 0)
att_sparse <- test1[,keep_var]

#analysis of reduced (but no NA) attributes data
#plotting a correlation heatmap
library(ggplot2)
total_cor <- cor(na.omit(att_sparse[c(2:24, 26)]))
c <- qplot(x=Var1, y=Var2, data=melt(total_cor), fill=value, geom="tile") 
c <- c + scale_fill_gradientn(colors = c("blue", "white","#cc0000"), limits = c(-1,1))
c <- c + theme(axis.text.x = element_text(angle = 35, hjust = 1))
c <- c + labs(fill = "Correlation")
c <- c+ ggtitle("Correlation Heatmap")
c <- c+  theme(axis.title.x=element_blank(), axis.title.y=element_blank())
c <- c + theme(plot.margin = unit(c(-0.9,0.5,4,0.5), "cm"))
#c <- c + geom_text(aes(Var2, Var1, label = format(round(value, 2), nsmall = 2)), color = "black", size = 4)
print(c)

#Performing linear regression on binary attributes (not including review count)
#note: performing regression on the 'sparse' (ie no NAs) dataset
att_ls <- lm(stars~., data = att_sparse[c(2:24)])
summary(att_ls)

#linear regression including review count
att_ls_count <- lm(stars~., data = att_sparse[c(2:24, 26)])
summary(att_ls_count)


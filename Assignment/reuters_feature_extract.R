reutdata_train = read.csv("reuters_train.csv",header = FALSE)
names(reutdata_train) = c("Document","Class","Type")
reutdata_test = read.csv("reuters_test.csv", header = FALSE)
names(reutdata_test) = c("Document","Class","Type")
reutdata_train$Class = as.character(reutdata_train$Class)
reutdata_train$Document = as.character(reutdata_train$Document)
str(reutdata_train)
reutfinal = subset(reutdata_train, reutdata_train[,2]!="unknown")
reutfinal$Class = as.factor(reutfinal$Class)
str(reutfinal$Class)
reutfinal$type = "train"
reutdata_test$type = "test"
reutdata_test$Document = as.character(reutdata_test$Document)
reutwhole = rbind(reutfinal, reutdata_test)
library(tm)
reutcorpus = Corpus(VectorSource(reutwhole$Document))
# creating a corpus# Refining
reutcorpus = tm_map(reutcorpus, tolower)                 # changing all the texts to lower case
reutcorpus = tm_map(reutcorpus, removePunctuation)       # removing punctuations from the text
reutcorpus = tm_map(reutcorpus, removeNumbers)           # removing numbers from the text
reutcorpus = tm_map(reutcorpus, removeWords, stopwords("english")) # removing english stopwords like articles, prepositions etc. from the text 
reutcorpus = tm_map(reutcorpus, stripWhitespace)   
# removing blank spaces
reutcorpus = tm_map(reutcorpus, stemDocument)
# stemming the text using porter's stemming algorithm
# Counting words
termdocument = DocumentTermMatrix(reutcorpus, control = list(weighting = weightTfIdf))
# Create a Document�Term�Matrix that counts individual words
termdocument_resized = removeSparseTerms(termdocument, .98) # Keeping only those words that appear more than 97% in the text
termdocument_resized$dimnames$Terms
# Viewing the words that are counted
# Feature building
reutwords = as.data.frame(as.matrix(termdocument_resized)) # Convert the DocumentTermMatrix to make new features
reutwhole = cbind(reutwhole, reutwords)
# merging the two matrices
reutwhole = reutwhole[,-1]
# Removing the message column from the matrix
names(reutwhole)
# features for classification
reuttrain = subset(reutwhole, reutwhole[,2]=="train")
reuttest = subset(reutwhole, reutwhole[,2]=="test")
reuttrain = reuttrain[,-2]
reuttrain_removed = reuttrain[-1,]
reuttest = reuttest[,-2]
write.table(reuttest, file = "reut_test_features.csv",quote = FALSE,col.names = FALSE,sep = ",")
#Classification
#Random Forest
rfFit <� randomForest(Class ~ .,
                      data = reuttrain,
                      ntree = 500,
                      mtry = 24,
                      importance = TRUE)
rfFit
# formula
# data set
# number of trees
# variables for split
# importance recorded
predTrain <� predict(rfFit, reuttrain, type = "class")
# prediction on train set
mean(predTrain == reuttrain$Class)
# classification accuracy
predValid <� predict(rfFit, reuttest, type = "class")
# prediction on validation set
mean(predValid == reuttest$Class)
# classification accuracy
importance(rfFit)
varImpPlot(rfFit)
# importance of the variables in the model (values)
# importance of the variables in the model (visual)
library("e1071")
# #Naive Bayes
# reutX = reuttrain[,�1]
# reutY = reutfinal[,1]
# reutY = as.vector(reutY)
# naiveFit = naiveBayes(as.matrix(reutX), # A numeric matrix, or a data frame of categorical and/or
numeric variables.
#
reutY,
# Class vector.
#
Class ~.,
# formula for fit
#
data = reuttrain) # dataset for fit
# pred <� predict(naiveFit, reuttest,type = "class")
# str(pred)
# mean(pred == reuttest$Class)
# table(pred, reutfinal$Class)
#Support Vector Machines
#Linear KernelsvmFit <� svm(Class ~ .,
data = reuttrain,
kernel = "linear",
cost = 1e7,
scale = FALSE)
# formula for fit
# dataset for fit
# choose a kernel
# relaxation cost
# feature�scaling
summary(svmFit)
# summary of the fitted model
predTrain = predict(svmFit, newdata = reuttrain)
mean(predTrain == reuttrain$Class)
predValid <� predict(svmFit, newdata = reuttest)
# prediction
mean(predValid == reuttest$Class)
#Radial Kernel
svmFit <� svm(Class ~ .,
              data = reuttrain,
              kernel = "radial",
              gamma = 1e3,
              cost = 1e6,
              scale = FALSE)
# formula for fit
# dataset for fit
# choose a kernel
summary(svmFit)
# relaxation cost
# feature�scaling
# summary of the fitted model
predTrain = predict(svmFit, newdata = reuttrain)
mean(predTrain == reuttrain$Class)
predValid <� predict(svmFit, newdata = reuttest)
mean(predValid == reuttest$Class)
# Predict the classes for the validation set
predValid <� predict(radFit, newdata = reuttest)
mean(predValid == reuttest$Class)
#Polynomial Kernel
svmFit <� svm(Class ~ .,
              #
              data = reuttrain,
              kernel = "polynomial",
              degree = 2,
              gamma = 1e3,
              coef0 = 0.01,
              cost = 1e6,
              scale = FALSE)
summary(svmFit)
# relaxation cost
# feature�scaling
# summary of the fitted model
library("gbm")
gbmFit <� gbm(Class ~ .,
              data = reutfinal,
              distribution = "multinomial",
              n.trees = 1000,
              interaction.depth = 7,
              shrinkage = 0.001,
              cv.folds = 1,
              n.cores = 8)
gbmFit
summary(gbmFit)
DEADLINE: 23/05/2017 11:59 IST
# prediction
formula for fit
# dataset for fit
# choose a kernel
predTrain = predict(svmFit, newdata = reuttrain)
mean(predTrain == reuttrain$Class)
predValid <� predict(svmFit, newdata = reuttest)
mean(predValid == reuttest$Class)
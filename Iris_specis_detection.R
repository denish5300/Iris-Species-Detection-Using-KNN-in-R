#Apply the kNN algorithm on the training set to detect Iris species of the test dataset.

#Inputs: Iris dataset
#Output file: Display values on the Console 
#==============================================================================

#1.	Read-in the "training" set and test set.
iris_train_all <- read.csv("iris_training.csv")
iris_test <- read.csv("iris_test.csv")

#Explore the data
str(iris_train_all)
str(iris_test)

#Remove the id feature in train and test dataset
iris_train_all <- iris_train_all[-1]
iris_test <- iris_test[-1]

#Class variables (the outcome we hope to predict): Species
table(iris_train_all$Species)

#Percentage of each case
round(prop.table(table(iris_train_all$Species)) * 100, digits = 2)

#Recode the training Species variable
iris_train_all$Species <- factor(iris_train_all$Species,  levels = c("setosa", "versicolor", "virginica"))

#Explore the features of the training data
summary(iris_train_all)

#Transformation - normalizing numeric data

#Create normalize function
normalize <- function(x){
  return((x - min(x))/(max(x) - min(x)))
}

#Apply transformation on data and explore the features of new data
iris_train_all_n <- as.data.frame(lapply(iris_train_all[1:4], normalize))
iris_test_n <- as.data.frame(lapply(iris_test, normalize))

summary(iris_train_all_n)
summary(iris_test_n)

#Create the training and validation set from original transformed "training" set
set.seed(02468)
rowvec <- 1:nrow(iris_train_all_n)

#Split train and validation set by ~80 and ~20 percentage randomly
train_vec <- sample(rowvec, floor(nrow(iris_train_all_n)* 80/100), replace = FALSE)
validation_vec <- rowvec[!is.element(rowvec, train_vec)]

iris_train_n <- iris_train_all_n[train_vec,]
iris_validation_n <-  iris_train_all_n[validation_vec,]

# Store class labels in factor vectors
iris_train_labels <- iris_train_all[train_vec,5]
iris_validation_labels <- iris_train_all[validation_vec,5]

#2.	Apply kNN .

#predict the class for the validation set with k = 11 {~sqrt of total number of data in training set}
k=11
iris_val_pred <- knn(train = iris_train_n, test = iris_validation_n, cl = iris_train_labels, k = k)

#Evaluate the model Performance by cross checking with validation set
CrossTable(x = iris_validation_labels, y = iris_val_pred,
           prop.chisq = FALSE,
           k = 11,
           prop.r = FALSE,
           prop.c = FALSE)

#Analysis: All validation data were correctly identified by the model when k =11. 


#Try different values of k (Sensitivity Analysis)
k_Values <- c(1, 5, 15, 21, 25, 31, 35, 41)
for (k in k_Values){
  iris_val_pred <- knn(train = iris_train_n, test = iris_validation_n, cl = iris_train_labels, k = k)
  message("++++++++++++++++ k=", k, " ++++++++++++++++")
  CrossTable(x = iris_validation_labels, y = iris_val_pred,
             prop.r=FALSE,
             prop.c = FALSE,
             prop.chisq=FALSE)
}
#Analysis: Only one data was incorrectly identified by the model when k >= 35.
#We chose k = 11 because it is not taking way too less and way too high number of neighbors into consideration
#while making the prediction. 

#3 predict the class for the test set with k = 11
k=11
iris_test_pred <- knn(train = iris_train_n, test = iris_test_n, cl = iris_train_labels, k = k)


#4.	Store the item IDs and class labels in a CSV file.
#Get test id from original Iris test data
iris_test <- read.csv("iris_test.csv")
iris_test_id <- iris_test[,1]
iris_test_pred <- data.frame("ID" = iris_test_id, "Species" = iris_test_pred)
write.csv(iris_test_pred, "C:/Users/Denishree/Desktop/MSDA/CS 5310 Data Mining/Week2 - KNN/Homework-1/iris_test_submission_Team3.csv", row.names = FALSE)






# Boston Housing Price Prediction with Neural Network ----------------------------

# Clean the System & Console Variable and Set the Path --------------------
rm(list = ls())
cat("\014")
options(warn = -1)

# Sys_Date <- format(Sys.Date(), "%Y%m%d")
# Sys_Time <- format(Sys.time(), "%H:%M:%S")


# Install & Use Library ---------------------------------------------------
if (require("MASS")==FALSE){
  install.packages("MASS")
  library(MASS)
}
if (require("neuralnet")==FALSE){
  install.packages("neuralnet")
  library(neuralnet)
}

# Set the System Path & Variable & Output Directory Name
if (require('here') == FALSE){
  install.packages('here')
  library(here)
}

Path <- here()

setwd(Path)

set.seed(123)

# Load Dataset
DataSet <- MASS::Boston

write.csv2(DataSet, file = "./Outputs/1_BostonHousingPrice_OriginalDataSet.csv", row.names = FALSE)

# MetaData of Dataset
help(Boston)

# Dimension of Dataset
dim(DataSet)

# Structure of Dataset
str(DataSet) 

# sapply(DataSet, class)

# Summary of Dataset
summary(DataSet)

# First 5 observations of DataSet 
head(DataSet, 5)

# End 5 observations of DataSet 
tail(DataSet, 5)

# Histogram Grafh of "Medv" in Dataset
m_name  <- "Boston Housing Price"
x_name <- "medv"

jpeg(filename = paste0("./Outputs/Histogram of Boston Housing Price.jpg"), units="px")
hist(DataSet$medv,
     main = paste("Histogram of" , m_name),
     xlab = x_name)
dev.off()

# Awesome Functions (apply, sapply,lapply)
apply(DataSet, 2, range)

MaxValue <- apply(DataSet, 2, max)
MinValue <- apply(DataSet, 2, min)

# Min-Max Normalization
DataSetN <- as.data.frame(scale(DataSet, center = MinValue, scale = MaxValue-MinValue ))

# Random Observation Selection 
ind <- sample(1:nrow(DataSet),400)

# Set the Train DataSet (400 Observation)
TrainDF <- DataSetN[ind,]
write.csv2(TrainDF, file = "./Outputs/2_BostonHousingPrice_TrainDataSet.csv", row.names = TRUE)

# Set the Test DataSet (106 Observation)
TestDF  <- DataSetN[-ind,]
write.csv2(TestDF, file = "./Outputs/3_BostonHousingPrice_TestDataSet.csv", row.names = TRUE)


# All Variable
AllVars <- colnames(DataSet)
# AllVars <- colnames(DataSet[1:14])

# Independent Variable
PredictVars <- AllVars[!AllVars%in%"medv"]
# PredictVars<- colnames(DataSet[1:13])

PredictVars <- paste(PredictVars, collapse = "+")

ModelFormula <- as.formula(paste("medv~", PredictVars, collapse = "+"))

# Create the Neural Network Model
# Input Layer Size: 13 Independent Variable
# 2 Hidden Layer
# First Hidden Layer Size:4
# Second Hidden Layer Size:2
# Outpur Layer Size: 1

NNModel <- neuralnet(formula = ModelFormula, hidden= c(4,2), linear.output=T, data=TrainDF)

# Plot Grafh of Neural Networks Models

jpeg(filename = paste0("./Outputs/NeuralNetworkModel.jpg"), units="px")
plot(NNModel)
dev.off()

PredictValues <- compute(NNModel, TestDF[ , 1:13])

# str(PredictValues)

PredictValuesResult <- PredictValues$net.result

ActualValues <- (TestDF$medv)

Result <- data.frame(Actual=ActualValues, Predict=PredictValuesResult, Diff=ActualValues-PredictValuesResult)
write.csv2(Result, file = "./Outputs/4_ActualPredictedValues.csv", row.names = TRUE)

# Head Of Result
head(Result)

ColMean <- apply(Result, 2, mean)

MSE <- sum((PredictValuesResult - ActualValues)^2)/nrow(TestDF) #Mean Square Error

cat("Mean Square Error: ", MSE)

# Plot of Actual Values and Predicted Values
jpeg(filename = paste0("./Outputs/ActualvsPredictedValues.jpg"), units="px")
plot(ActualValues, PredictValuesResult, col='blue', 
     main='Actual vs Predicted', pch=1, cex=1, type = "p", 
     xlab ="Actual", ylab="Predict")
abline(0,1,col="red")
dev.off()

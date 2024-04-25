install.packages("randomForest")
library(randomForest)
library(caret)
library(ROCR)

iris_df <- read.csv("iris.csv")

set.seed(42)
trainIndex <- createDataPartition(iris_df$target, p = 0.8, list = FALSE)
train_data <- iris_df[trainIndex,]
test_data <- iris_df[-trainIndex,]

clf <- randomForest(target ~ ., data = train_data, ntree = 100)

y_pred <- predict(clf, test_data)

accuracy <- mean(y_pred == test_data$target)
cat("Test seti doğruluğu:", accuracy * 100, "%\n")

cm <- table(test_data$target, y_pred)
print("Karmaşıklık Matrisi:")
print(cm)

class_report <- confusionMatrix(y_pred, test_data$target)
print("Sınıflandırma Raporu:")
print(class_report)

pred_prob <- predict(clf, test_data, type = "prob")
pred_prob <- as.data.frame(pred_prob)

pred_prob$target <- test_data$target

roc_data <- prediction(pred_prob[,1:3], labels = as.factor(pred_prob$target))

roc_perf <- performance(roc_data, "tpr", "fpr")

plot(roc_perf, col=c("blue", "red", "green"), lwd=2,
     main = "ROC Curve",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")
legend("bottomright", legend = c("Class 0", "Class 1", "Class 2"), col = c("blue", "red", "green"), lty = 1, cex = 0.8)

new_flower <- data.frame(Sepal.Length = 5.1, Sepal.Width = 3.5, Petal.Length = 1.4, Petal.Width = 0.2)
predicted_class <- predict(clf, new_flower)
cat("Tahmin edilen Iris türü:", predicted_class, "\n")
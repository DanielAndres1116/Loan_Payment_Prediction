# 1. Loan Payment Prediction: is this person going to pay on time?

Loan payment prediction models are created using machine learning techniques to predict the likelihood of a loan being repaid on time. These models analyze a variety of factors such as credit score, income, loan amount, and other relevant information to make predictions about loan repayment.

There are several types of machine learning algorithms that can be used for loan payment prediction, including decision trees, random forests, logistic regression, and neural networks. Each of these algorithms has its own strengths and weaknesses, and the choice of algorithm will depend on the specifics of the loan payment prediction problem being solved.

These loan payment prediction models are useful in many different businesses, including banking and financial services, consumer lending, and retail. For example, banks can use these models to predict loan repayment probabilities for individual borrowers, which can help them make more informed lending decisions. Similarly, consumer lending companies can use these models to predict loan repayment probabilities for their customers, which can help them make more informed decisions about loan origination and pricing.

In addition to helping companies make better lending decisions, loan payment prediction models can also be used to identify trends and patterns in loan repayment data. This information can be used to improve lending practices, reduce the risk of loan defaults, and ultimately improve overall loan portfolio performance.
Overall, loan payment prediction models created using machine learning techniques are a powerful tool for businesses in the lending and financial services industry. By providing accurate predictions about loan repayment probabilities, these models can help companies make more informed decisions and improve their bottom line.

## 1.1. Dataset Description

This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes fields as: Loan Status, Principal Loan Amount, Terms, Effective Date, Due Date, Age of applicant, Education of applicant and Gender of applicant. There are two files, one for the train set and other for the test set.

## 1.2. Project Description

This project is about building four Machine Learning Models aimed to classification problems that are able to make predictions about loan payments, these models are K-Nearest Neighbor (KNN), Decision Tree, Support Vector Machine and Logistic Regression. And we’ll compare them each other’s results.

•	K-Nearest Neighbor (KNN) is a simple and popular machine learning algorithm used for classification and regression tasks. It works by finding the K closest data points (neighbors) to a new data point and classifying or predicting its target variable based on the majority of the K nearest data points. KNN uses a distance metric such as Euclidean distance to determine the proximity of data points to one another. The value of K is set by the user and it can be any positive integer. The larger the value of K, the smoother the decision boundary will be, but the greater the chance of misclassifying points.
•	Decision Tree Classification is a type of machine learning algorithm used for classification problems. It uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. At each internal node of the tree, a decision or test is performed to assign a case to one of the branches (sub-trees) leading from that node, until a leaf node (terminal node) is reached that assigns the case to a class. The decisions are based on the values of the input variables, and the goal is to find the tree structure that correctly classifies the largest number of cases in the training data. The results of a decision tree can be easily interpreted and explained, making it a popular algorithm for use in many practical applications.
•	Support Vector Machine (SVM) is a type of supervised machine learning algorithm used for classification or regression analysis. It uses the concept of “maximum margin” to find the hyperplane (a decision boundary) that separates the data into different classes. This hyperplane is chosen such that it maximizes the margin between the closest data points of different classes, known as support vectors. SVM is particularly useful when the data is non-linearly separable, as it can apply a non-linear transformation to the data to find a suitable boundary. SVM is widely used in real-world applications, such as text classification, image recognition, and bioinformatics.
•	Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. In the context of binary classification, the outcome is a binary variable, which can take on one of two possible values, such as “yes” or “no,” “true” or “false,” or “1” or “0.” Logistic Regression models are used to predict a binary outcome by estimating the probability of the positive class. The model uses a logistic function to map the input variables to a probability value between 0 and 1, which can then be thresholded to produce the binary outcome. Logistic Regression models are widely used in a variety of applications, such as medical diagnosis, credit risk assessment, and marketing.

Previous to the model, was needed a preprocessing of the data for the training set as the testing set. All the process is well described on the notebook. Following, I’ll explain the programming code.

## 1.3. Programming Code Explanation

This is a script in Python using several libraries (itertools, numpy, matplotlib, pandas, seaborn and scikit-learn) to build a machine learning model to predict the loan status of applicants based on several features such as Principal, terms, age, gender, education, and day of the week.

The script begins by loading a data file (loan_train.csv) into a Pandas dataframe df. It then converts the columns due_date and effective_date into datetime format and prints the value count of the loan_status column.

Next, the script creates histograms for the Principal, age, and dayofweek columns, grouping the data by Gender and coloring the bars by loan_status. These histograms are visualized using the seaborn library and its FacetGrid class.

The script then creates a new column weekend based on the dayofweek column and normalizes the value counts of the loan_status column based on Gender and education. The data is then preprocessed by scaling the features to have a mean of 0 and a standard deviation of 1. The script splits the preprocessed data into training and testing sets.

Finally, the script builds and evaluates two machine learning models: a k-Nearest Neighbor (kNN) classifier and a Decision Tree classifier. For the kNN model, it first trains a model with k=3 and then uses a loop to find the best value of k based on accuracy. The best value of k is found to be 7. The script then trains a final kNN model with k=7. The Decision Tree model is trained and evaluated on the test set.

Overall, the script goes through the steps of loading and preprocessing the data, splitting it into training and testing sets, building and evaluating two machine learning models, and selecting the best model based on accuracy.

## 1.4. Result Analysis and Conclusions

We’ll see certain histograms created to analyze the information presented in the dataset and the potential features that can be useful to predict the mentioned target. These are histogram plots with multiple facets, also known as a faceted histogram. The histograms are split by the “Gender” column and the “loan_status” column is used to color code the bars in the histograms.

![image](https://user-images.githubusercontent.com/43154438/229928018-c6b84a74-f2d6-4405-a800-631018075390.png)

Figure 1: 

![image](https://user-images.githubusercontent.com/43154438/229928095-f8b50fe6-62c0-4418-9210-448422f79c61.png)

Figure 2: 

![image](https://user-images.githubusercontent.com/43154438/229928161-88823c86-e201-467f-b29c-dfbd79e21c8f.png)

Figure 3: 

![image](https://user-images.githubusercontent.com/43154438/229928199-98fdcf31-8b20-4139-8062-2b5baeff7ae7.png)

Figure 4: 

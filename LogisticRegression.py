Author: Gabriel Larot
Date: Mar 4, 2024

import random
import math

class LogisticRegression:
    def __init__(self):
        self.__rate = 0.01
        self.__weights = []
        self.__ITERATIONS = 200

    @property
    def mWeights(self):
        return self.__weights

    # Implement the sigmoid function
    def sigmoid(self, l_combination):
        return 1 / (1 + (math.exp(-1 * l_combination)))

    # Helper function for prediction
    # Takes a test instance as input and outputs the probability of the label being 1
    def predictHelper(self, test_instance):
        pass_to_sig = sum([weight*int(feature) for weight, feature in zip(self.__weights, test_instance[:-1])])
        predicted_label = self.sigmoid(pass_to_sig)

        if predicted_label > 0.5:
            return 1
        else:
            return 0

    # Prediction function
    # Takes a test instance as input and outputs the predicted label
    # Calls the Helper function
    def predict(self, test_instance):
        return self.predictHelper(test_instance)

    # Takes a test set as input, call the predict function to predict a label for it,
    # and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix
    def predictLabels(self, test_data):
        # test_data has 1115 observations and 1364 features, 1 class label
        #       print(len(test_data))
        #       print(len(test_data[0]))

        predicted_labels = []
        for i, j in test_data.items():
            predicted_labels.append(self.predict(j))

        # logic for the confusion matrix
        #   2x2 array, for the predicted class and the actual class: [[0, 0], [0, 0]]
        #   if both are yes (1):                            inc tp -> x in [[x, 0], [0, 0]]
        #   if predicted is no (0), but actual is yes (1):  inc fn -> x in [[0, x], [0, 0]] 
        #   if predicted is yes (1), but actual is no (0):  inc fp -> x in [[0, 0], [x, 0]] 
        #   if both are no (0):                             inc tn -> x in [[0, 0], [0, x]] 
  
        confusion_matrix = [[0, 0], [0, 0]]
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1 and int(test_data[i][-1]) == 1:
                confusion_matrix[0][0] += 1
            elif predicted_labels[i] == 0 and int(test_data[i][-1]) == 1:
                confusion_matrix[0][1] += 1
            elif predicted_labels[i] == 1 and int(test_data[i][-1]) == 0:
                confusion_matrix[1][0] += 1
            elif predicted_labels[i] == 0 and int(test_data[i][-1]) == 0:
                confusion_matrix[1][1] += 1

        # Print the confusion matrix
        confusion_matrix_flag = 0
        if confusion_matrix_flag:
            print()
            for i in range(len(confusion_matrix)):
                for j in range(len(confusion_matrix)):
                    print(confusion_matrix[i][j], end='\t')
                print()

        # True pos = tp, False neg = fn, False pos = fp, True neg = tn
        tp = confusion_matrix[0][0]
        fn = confusion_matrix[0][1]
        fp = confusion_matrix[1][0]
        tn = confusion_matrix[1][1]

        # Accuarcy calculation
        accuracy = round((tp + tn) / (tp + fn + fp + tn), 3)

        # Precision calculation
        precision = round(tp / (tp + fp), 3)

        # Recall calculation
        recall = round(tp / (tp + fn), 3)

        # F1 score calculation
        f1 = round((2 * recall * precision) / (recall + precision), 3)

        print(f'\nAccuracy: {accuracy}\
                \nPrecision: {precision}\
                \nRecall: {recall}\
                \nF1 score: {f1}')
 

    # Train the Logistic Regression in a function using Stochastic Gradient Descent
    # Also compute the log-loss in this function
    def trainLR(self, train_data):
        # set weights to 0 before training
        if not self.__weights:
            self.__weights = [0] * (len(train_data[0])-1)

        # Train with SGD
        sig = 0; sigmoids = []
        # iterate 200 times (num keys)
        for i in range(len(train_data.keys())):
            # Sum the linear combination of the weights and features: (B_1*x_1 + B_2*x_2 + ... + B_n*x_n)
            pass_to_sig = sum([weight*int(feature) for weight, feature in zip(self.__weights, train_data[i][:-1])])
            sig = self.sigmoid(pass_to_sig)
        
            # for log_loss calc
            sigmoids.append(sig)

            # Iterate over the length of the weights (1364 elements)
            for i2 in range(len(self.__weights)):
                # weight_new       =  weight_old         - [learn_rate  * ((sig -    actual_y^(i) )      *     x_j^(i) ))]
                self.__weights[i2] = (self.__weights[i2] - (self.__rate * ((sig - int(train_data[i][-1])) * int(train_data[i][i2]))))

        # Checking weights (for visual use only - delete later)
        weight_flag = 0
        if weight_flag:
            count = 1; pos = 0; neg = 0; zero = 0
            for i in range(len(self.__weights)):
                if not count % 10 and count <= 100:
                    print('{:.4f}'.format(round(self.__weights[i], 4)))
                elif self.__weights[i] >= 0 and count <= 100:
                    print('{:.4f}'.format(round(self.__weights[i], 4)), end='  | ')
                elif self.__weights[i] < 0 and count <= 100:
                    print('{:.4f}'.format(round(self.__weights[i], 4)), end=' | ')
                count += 1

                # Checking number of pos, neg, and zero weights
                if self.__weights[i] > 0: pos += 1
                elif self.__weights[i] == 0: zero += 1
                elif self.__weights[i] < 0: neg += 1
            print(f'\npercent pos: {round(pos/len(self.__weights), 2)}\
                \tpercent neg: {round(neg/len(self.__weights), 2)}\
                \tpercent zero: {round(zero/len(self.__weights), 2)}')
        
        # Compute the log-loss/Cross-Entropy Loss
        # Minimize J(theta) = - sum(from i=1 to n) [ y_i * log(h_theta(x_i)) + (1-y_i) * log(1 - h_theta(x_i))]
        log_loss = 0
        for i, j in train_data.items():
            # Use the weights as they were getting updated from sgd
            log_loss_flag = 0
            if log_loss_flag:
                log_loss += ( ((int(j[-1]) * math.log(sigmoids[i]))) + ((1 - int(j[-1])) * math.log(1-sigmoids[i])) )
            
            # or should i use the final weights from sgd to pass to the sigmoid function with each feature?
            else:
                pass_to_sig = sum([weight*int(feature) for weight, feature in zip(self.__weights, train_data[i][:-1])])
                sig = self.sigmoid(pass_to_sig)
                log_loss += ( ((int(j[-1]) * math.log(sig))) + ((1 - int(j[-1])) * math.log(1-sig)) )
                
        log_loss *= (-1/ (2 * len(train_data)))
        print(f'\nLog loss: {log_loss}') 
        
    # Read the input dataset
    def readData(self, path, train_or_test):
        # FIXME - randomly select the lines? might need to iterate through the dataset multiple times?
        count = 0
        my_dict = {}
        if train_or_test == 'train':
            with open(path, 'r') as my_file:
                for line in my_file:
                    if count > 0:
                        my_dict[count-1] = line.strip('\n').split(',')
                    count += 1

            self.trainLR(my_dict)
            
        elif train_or_test == 'test':
            with open(path, 'r') as my_file:
                for line in my_file:
                    if count > 0:
                        my_dict[count-1] = line.strip('\n').split(',')
                    count += 1
            self.predictLabels(my_dict)
        
def main():
    lr = LogisticRegression()
    
    home_computer = 1; 
    if home_computer:
        train_file_path = r'C:\Users\Gabe\OneDrive\Desktop\asgn1data\train-1.csv'
        test_file_path = r'C:\Users\Gabe\OneDrive\Desktop\asgn1data\test-1.csv'
    else:
        train_file_path = '/Users/gabriellarot/Desktop/SJSU/Spring \'24/CS-171/Programming Assignment 1/Data/train-1.csv'
        test_file_path = '/Users/gabriellarot/Desktop/SJSU/Spring \'24/CS-171/Programming Assignment 1/Data/test-1.csv'
    
    lr.readData(train_file_path, train_or_test='train')
    lr.readData(test_file_path, train_or_test='test')


if __name__ == '__main__':
    main()

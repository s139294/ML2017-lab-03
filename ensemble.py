#coding: utf-8
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_clf = weak_classifier
        self.n_weakers_limit = n_weakers_limit

        #sample weights
        self.w = None
        #error rate: classifier weights
        self.alpha_list = []
        
        self.error_list = [] #list for storing error rate during iteration
        self.weak_clf_list = [] #list for storing weak base classifiers during iteration

        self.validation_score_list = []

    def fit(self,X,y, X_val=None, y_val=None):
        '''Build a boosted classifier from the training set (X, y).
        
        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
            validation set(X_val, y_val): Used for early stopping.
        '''
        #initialize sample weights
        self.w = np.ones(X.shape[0])/X.shape[0]

        for i in range(self.n_weakers_limit):
            cur_clf = self.weak_clf.fit(X, y, sample_weight=self.w)
            predict_res = cur_clf.predict(X)
            cur_error = np.sum((predict_res != y.reshape(-1,)) * self.w)
            print(i,'base clf training accuracy: ', np.mean(y.reshape(-1,) == predict_res))
            if cur_error > 0.5:#base classifer is not good enough, discard it
                continue
            elif cur_error == 0:
                print('This base classifer has training error equals 0. End of iteration.')
                break

            self.weak_clf_list.append(cur_clf)
            self.error_list.append(cur_error)

            cur_alpha = 1/2 * np.log((1-cur_error)/cur_error)
            self.alpha_list.append(cur_alpha)

            self.w *= np.exp(-cur_alpha * y.reshape(-1,) * predict_res)
            self.w /= np.sum(self.w)

            if (X_val is not None) & (y_val is not None):
                res = np.mean(self.predict(X_val)==y_val)
                self.validation_score_list.append(res)
                print('current validation accuracy: ', res)

    def predict_scores(self, X, use_first_k_clfs):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        res = np.zeros((X.shape[0]))

        k = len(self.weak_clf_list)
        if use_first_k_clfs >= 0:
            k = use_first_k_clfs
        for i in range(k):
            res += self.weak_clf_list[i].predict(X) * self.alpha_list[i]
        return res

    def predict(self, X, threshold=0, use_first_k_clfs=-1):#可以选择使用前k棵决策树进行预测（根据验证集的结果）
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        res = self.predict_scores(X, use_first_k_clfs)
        res[res >= threshold]=1
        res[res < threshold]=-1
        return res.reshape(-1,1)

    def plotting(self):
        plt.title('Adaboost')
        plt.xlabel('iteration number')
        plt.ylabel('validation accuracy')
        plt.plot(range(len(self.validation_score_list)), self.validation_score_list)  
        plt.grid()
        plt.show()

    def get_report(self, X_test, y_test):
        #get the index of the highest value in validation_score_list
        idx = self.validation_score_list.index(max(self.validation_score_list))
        pred_res = self.predict(X_test, use_first_k_clfs=-1)
        with open('report.txt', "wb") as f:
            repo=classification_report(y_test,pred_res,target_names=["face","nonface"])
            f.write(repo.encode())

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

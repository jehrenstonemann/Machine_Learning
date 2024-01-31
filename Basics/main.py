# project: p5
# submitter: xhuang438
# partner: none
# hours: 4

import pandas as pd
import sklearn.linear_model

class UserPredictor:
    
    def __init__(self):
        self.model = sklearn.linear_model.LogisticRegression()
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
    
    def fit(self, train_users, train_logs, train_y):
        train_users = train_users.replace("gold",1)
        train_users = train_users.replace("silver",2)
        train_users = train_users.replace("bronze",3)
        train_y["y"] = train_y["y"].astype(int)
        
        train_users["Total Seconds"] = train_logs.groupby("user_id").sum()
        train_users["Total Seconds"] = train_users["Total Seconds"].fillna(0)
        train_users["Total Seconds"] = train_users["Total Seconds"].astype('int')
        
        self.x_train = train_users.iloc[:, 2:6]
        self.y_train = train_y.iloc[:, 1:2]
        
        self.model.fit(self.x_train, self.y_train.values.ravel())
        
    def predict(self, test_users, test_logs): 
        test_users = test_users.replace("gold",1)
        test_users = test_users.replace("silver",2)
        test_users = test_users.replace("bronze",3)

        test = test_logs.groupby("user_id")['seconds'].sum().reset_index()
        test = test.rename(columns={'seconds': 'Total Seconds'})
        test_users = test_users.merge(test, on='user_id', how='left')
        test_users["Total Seconds"] = test_users["Total Seconds"].fillna(0)
        test_users["Total Seconds"] = test_users["Total Seconds"].astype('int')
        
        x_test = test_users.iloc[:,2:6]
        
        return self.model.predict(x_test)
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_validate
from sklearn.neural_network import MLPRegressor

if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"

    def get_train_post_data():
        rough_data = open("posts_train.txt", "r")
        next(rough_data) #this line skips the header in the file
  
        user_dict = {}
        for line in rough_data:
            #Parse data in each line
            split = line.split(",")
            user_id, hr_1, hr_2, hr_3, lat, lon, posts = int(split[0]), int(split[1]), int(split[2]), int(split[3]), float(split[4]), float(split[5]), int(split[6])
            if (hr_2 == 25):
                hr_2 = hr_1
            if (hr_3 == 25):
                hr_3 = hr_2
            if (not(lat == 0 and lon == 0)):
                user = create_dict(hr_1, hr_2, hr_3, posts, lat, lon)
                user_dict[user_id] = user
        return user_dict

    def get_test_post_data(user_dict):
        #given user dict that already has train post data, get info for test post data!
        rough_data = open("posts_test.txt", "r")
        next(rough_data) #this line skips the header in the file

        for line in rough_data:
            #Parse data in each line
            split = line.split(",")
            user_id, hr_1, hr_2, hr_3, posts = int(split[0]), int(split[1]), int(split[2]), int(split[3]), int(split[4])
            if (hr_2 == 25):
                hr_2 = hr_1
            if (hr_3 == 25):
                hr_3 = hr_2
            user = create_dict(hr_1, hr_2, hr_3, posts, 0, 0, test = True)
            user_dict[user_id] = user
        return user_dict

    def create_dict(hr_1, hr_2, hr_3, posts, lat, lon, test = False):
        if (test):
            return {"hr_1" : hr_1, "hr_2" : hr_2, "hr_3": hr_3, "num_posts" : posts, "num_friends" : 0, "friends_lat" : 0, "friends_lon" : 0, "fr_friends_lat" : 0, "fr_friends_lon" : 0, "friends" : [], "test" : True}
        else:
            return {"hr_1" : hr_1, "hr_2" : hr_2, "hr_3": hr_3, "num_posts" : posts, "num_friends" : 0, "lat" : lat, "lon" : lon, "friends_lat" : 0, "friends_lon" : 0, "fr_friends_lat" : 0, "fr_friends_lon" : 0, "friends" : [], "test" : False}

    def get_user_dict():
        #gets the post data by calling get_post_data, then uses the graph data to generate features
        #returns a dictionary mapping users to features
        train_dict = get_train_post_data()
        user_dict = get_test_post_data(train_dict)
        
        graph = open("graph.txt")
        
        previous = 0
        friends_locations_calculated = 0 #used to compute average lat/lon of friends
        friends_hours_calculated = 0
        num_friends_posts = 0 #used to compute weighted average (by num of posts) of lat/lon of friends
        hours = [] #array where each index represents an utc hour. Used to find modal hour of friends' posts
        for i in range(26):
            hours.append(0)
        for line in graph:
            edge = line.split("	")
            current = int(edge[0])
            friend = int(edge[1])
            if(previous != 0 and previous != current):
                if(friends_locations_calculated > 0):
                    user_dict[previous]["friends_lat"] /= friends_locations_calculated
                    user_dict[previous]["friends_lon"] /= friends_locations_calculated
                friends_locations_calculated = 0
                
            #if user is one for which we have data, count number of friends
            if(current in user_dict):
                #increment num_friends by one
                user_dict[current]["num_friends"] += 1
                if (friend in user_dict):
                    user_dict[current]["friends"].append(friend)
                    if(user_dict[friend]["test"] == False):
                        #if have info on friends lat/lon, use it to calculated avg friend lat/lon
                        if(user_dict[friend]["lat"] != 0 and user_dict[friend]["lon"] != 0):
                            #only do it if friend is not in null island
                            user_dict[current]["friends_lat"] += user_dict[friend]["lat"]
                            user_dict[current]["friends_lon"] += user_dict[friend]["lon"]
                            friends_locations_calculated += 1                    
            previous = current
        
        for user, sub_dict in user_dict.items():
            #now go through users and calculate the average lat/lon of their friends' friends
            friends_calculated = 0
            for i in range(len(sub_dict["friends"])):
                #for each friend
                friend = sub_dict["friends"][i]
                friend_dict = user_dict[friend]
                for j in range(len(friend_dict["friends"])):
                    #for each friend's friends
                    fr_friend = friend_dict["friends"][j]
                    if (fr_friend in user_dict and fr_friend != user and user_dict[fr_friend]["test"] == False and user_dict[fr_friend]["lat"] != 0):
                        # if friend's friend exists in dictionaty, if have info on friend's friends,
                        # friends' friend is not the user you are calculating this for (avoid overfitting)
                        # and friend's friends is not in null island, use it to caculate avg lat/lon of friends' friends
                        sub_dict["fr_friends_lat"] += user_dict[fr_friend]["lat"]
                        sub_dict["fr_friends_lon"] += user_dict[fr_friend]["lon"]
                        friends_calculated += 1
            if (friends_calculated > 0):
                sub_dict["fr_friends_lat"] /= friends_calculated
                sub_dict["fr_friends_lon"] /= friends_calculated
                
        return user_dict

    def get_data(user_dict, lat = True):
        #given a dictionary with users and respective features, use it to generate train and test data.
        Xs_tr = []
        Xs_te = []
        y_tr = []
        IDs = []
        for key, sub_dict in user_dict.items():
            if (lat):
                #if predicting latitude
                X = np.array([sub_dict["friends_lat"], sub_dict["friends_lon"]])
            else:
                #if predicting longitude
                X = np.array([sub_dict["hr_1"], sub_dict["hr_2"], sub_dict["hr_3"], sub_dict["num_friends"], sub_dict["num_posts"], sub_dict["fr_friends_lat"], sub_dict["fr_friends_lon"], sub_dict["friends_lat"], sub_dict["friends_lon"]])
            if (sub_dict["test"] == True):
                #if on test set, append data to test design matrix and add user id to list of ids
                Xs_te.append(X)
                IDs.append(key)
            else:
                y = np.array([sub_dict["lat"], sub_dict["lon"]])
                Xs_tr.append(X)
                y_tr.append(y)
        X_te_array = np.array(Xs_te, dtype = float)
        X_tr_array = np.array(Xs_tr, dtype = float)
        y_tr_array = np.array(y_tr, dtype = float)
        return X_tr_array, X_te_array, y_tr_array, IDs

    def predict_test(lat_learner, lon_learner):
        #given a learner, use it to make predictions on test set
        #fit and predict for latitude
        user_dict = get_user_dict()
            
        X_tr, X_te, y, user_ids = get_data(user_dict, lat = True)
        lat = np.transpose(y)[0]
        lon = np.transpose(y)[1]

        #preprocess data by scaling to range [0, 1]
        scaler = sklearn.preprocessing.MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        lat_learner.fit(X_tr, lat)

        X_te = scaler.fit_transform(X_te)
        lat_preds = lat_learner.predict(X_te)

        #now fit and predict for longitude
        X_tr, X_te, y, user_ids = get_data(user_dict, lat = False)

        #preprocess data by scaling to range [0, 1]
        X_tr = scaler.fit_transform(X_tr)
        lon_learner.fit(X_tr, lon)

        X_te = scaler.fit_transform(X_te)
        lon_preds = lon_learner.predict(X_te)

        predictions = open("predictions.txt", "w")
        predictions.write("Id,Lat,Lon\n")
        for i in range(X_te.shape[0]):
            predictions.write(str(user_ids[i]) + "," +  str(round(lat_preds[i], 3)) + "," + str(round(lon_preds[i], 3)) + "\n")

        plt.clf
        plot_2 = plt.figure()
        axis = plot_2.add_subplot(1, 1, 1)
        axis.scatter(lon_preds, lat_preds)

        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.savefig("preds")


    knn = KNeighborsRegressor(n_neighbors = 50, p = 2)
    neural = MLPRegressor((50, 25, 5), max_iter = 1000)
    predict_test(knn, neural)s
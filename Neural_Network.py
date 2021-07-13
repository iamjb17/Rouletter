import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NN:
    history = None
    model = None
    df_data = None
    df_blind_test_data = None
    x = None
    y = None
    x_blind_test = None
    y_blind_test = None
    x_train, x_test, y_train, y_test = None, None, None, None
    x_train_scaled, x_test_scaled, x_blind_test_scaled = None, None, None
    y_predictions = None
    y_blind_test_predictions = None
    correlation = None
    figure_1 = None
    figure_2 = None
    figure_3 = None
    log_view = []

    figure = None
    canvas = None
    toolbar = None

    mse_nn, mae_nn = None, None

    # Load main data into dataframe and confirm
    def loadData(self):
        self.df_data = pd.read_csv('data/Capstone_data_Sheet1.csv')
        self.df_data.head()

        # Load blind test data into dataframe and confirm
        self.df_blind_test_data = pd.read_csv('data/Blind_Capstone_data_Sheet1.csv')
        self.df_blind_test_data.head()

    # ########## Data Transformations ##########
    def dataTransform(self):
        # Drop the columns that will not be used during training or testing
        self.df_data.drop('Ball Start Area', axis=1, inplace=True)
        self.df_data.drop('Zero Start Area', axis=1, inplace=True)
        self.df_data.drop('Ball Direction', axis=1, inplace=True)
        # df_data.drop('Zero Direction', axis=1, inplace=True)
        # df_data.drop('Ball2 ms/r', axis=1, inplace=True)
        # df_data.drop('Zero ms/r', axis=1, inplace=True)
        # df_data.drop('Zero Position3 /br', axis=1, inplace=True)
        # df_data.drop('Ball peg hit', axis=1, inplace=True)

        # Drop columns in the blind test df that will not be used
        self.df_blind_test_data.drop('Ball Start Area', axis=1, inplace=True)
        self.df_blind_test_data.drop('Zero Start Area', axis=1, inplace=True)
        self.df_blind_test_data.drop('Ball Direction', axis=1, inplace=True)
        # df_blind_test_data.drop('Zero Direction', axis=1, inplace=True)
        # df_blind_test_data.drop('Ball2 ms/r', axis=1, inplace=True)
        # df_blind_test_data.drop('Zero ms/r', axis=1, inplace=True)

        # Group neighbor numbers on the roulette wheel together. Either 12 groups, 6 groups, 3 groups
        # df_data.replace({0: 12, 1: 7, 2: 2, 3: 11, 4: 1, 5: 6, 6: 3, 7: 10, 8: 5, 9: 9, 10: 6, 11: 4, 12: 11, 13: 4, 14: 8, 15: 12, 16: 7, 17: 2, 18: 9, 19: 1, 20: 8, 21: 1, 22: 9, 23: 5, 24: 6, 25: 2, 26: 11, 27: 3, 28: 10, 29: 10, 30: 5, 31: 8, 32: 12, 33: 7, 34: 3, 35: 11, 36: 4}, inplace=True)
        # df_data.replace({0: 1, 1: 4, 2: 2, 3: 6, 4: 1, 5: 4, 6: 2, 7: 6, 8: 3, 9: 5, 10: 4, 11: 3, 12: 6, 13: 3, 14: 5, 15: 1, 16: 4, 17: 2, 18: 5, 19: 1, 20: 5, 21: 1, 22: 5, 23: 3, 24: 4, 25: 2, 26: 6, 27: 2, 28: 6, 29: 6, 30: 3, 31: 5, 32: 1, 33: 4, 34: 2, 35: 6, 36: 3}, inplace=True)
        self.df_data.replace(
            {0: 1, 1: 2, 2: 1, 3: 3, 4: 1, 5: 2, 6: 1, 7: 3, 8: 2, 9: 3, 10: 2, 11: 2, 12: 3, 13: 2, 14: 3, 15: 1,
             16: 2,
             17: 1,
             18: 3, 19: 1, 20: 3, 21: 1, 22: 3, 23: 2, 24: 2, 25: 1, 26: 3, 27: 1, 28: 3, 29: 3, 30: 2, 31: 3, 32: 1,
             33: 2,
             34: 1, 35: 3, 36: 2}, inplace=True)

        # Replace string values with appropriate number values. Model can only work with numbers
        self.df_data.replace({'CCW': 0, 'CW': 1}, inplace=True)
        self.df_data.replace({'North': 0, 'East': 1, 'South': 2, 'West': 3}, inplace=True)

        # Group neighbor numbers on the roulette wheel together. Either 12 groups, 6 groups, 3 groups
        # df_blind_test_data.replace({0: 12, 1: 7, 2: 2, 3: 11, 4: 1, 5: 6, 6: 3, 7: 10, 8: 5, 9: 9, 10: 6, 11: 4, 12: 11, 13: 4, 14: 8, 15: 12, 16: 7, 17: 2, 18: 9, 19: 1, 20: 8, 21: 1, 22: 9, 23: 5, 24: 6, 25: 2, 26: 11, 27: 3, 28: 10, 29: 10, 30: 5, 31: 8, 32: 12, 33: 7, 34: 3, 35: 11, 36: 4}, inplace=True)
        # df_blind_test_data.replace({0: 1, 1: 4, 2: 2, 3: 6, 4: 1, 5: 4, 6: 2, 7: 6, 8: 3, 9: 5, 10: 4, 11: 3, 12: 6, 13: 3, 14: 5, 15: 1, 16: 4, 17: 2, 18: 5, 19: 1, 20: 5, 21: 1, 22: 5, 23: 3, 24: 4, 25: 2, 26: 6, 27: 2, 28: 6, 29: 6, 30: 3, 31: 5, 32: 1, 33: 4, 34: 2, 35: 6, 36: 3}, inplace=True)
        self.df_blind_test_data.replace(
            {0: 1, 1: 2, 2: 1, 3: 3, 4: 1, 5: 2, 6: 1, 7: 3, 8: 2, 9: 3, 10: 2, 11: 2, 12: 3, 13: 2, 14: 3, 15: 1,
             16: 2,
             17: 1,
             18: 3, 19: 1, 20: 3, 21: 1, 22: 3, 23: 2, 24: 2, 25: 1, 26: 3, 27: 1, 28: 3, 29: 3, 30: 2, 31: 3, 32: 1,
             33: 2,
             34: 1, 35: 3, 36: 2}, inplace=True)

        # Replace string values with appropriate number values. Model can only work with numbers
        self.df_blind_test_data.replace({'CCW': 0, 'CW': 1}, inplace=True)

    # Define dependant(y) and independent(x) variables in main df and test df
    def defineXY(self):
        self.x = self.df_data.drop('Result Number', axis=1).values
        self.y = self.df_data['Result Number'].values

        self.x_blind_test = self.df_blind_test_data.drop('Result Number', axis=1).values
        self.y_blind_test = self.df_blind_test_data['Result Number'].values

        # Split main dataset into training and testing sets.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25,
                                                                                random_state=0)

    # Normalize the values of the dataframe by column to have the rows scale relative to each other
    def scaleValues(self):
        # Create a scaler to scale the data(Optional but optimal)
        scaler = StandardScaler()

        # fit the data using the scaler
        scaler.fit(self.x_train)
        # fit_transform(X))

        # scale the main data
        self.x_train_scaled = scaler.transform(self.x_train)
        self.x_test_scaled = scaler.transform(self.x_test)

        # scale the blind test data
        self.x_blind_test_scaled = scaler.transform(self.x_blind_test)
        # test_y_scaled = scaler.transform(test_y)

    # ######### Define the Model ############
    def buildModel(self):
        # Create model and add dense layers
        self.model = Sequential()
        self.model.add(Dense(192, input_dim=5, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        # Compile the model
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        self.model.summary()

        # Train the model for a fixed number of epochs
        self.history = self.model.fit(self.x_train_scaled, self.y_train, validation_split=0.2, epochs=150)

    # Pass the test data through the model and look at each prediction
    def prediction_test(self):
        self.y_predictions = self.model.predict(self.x_test_scaled)

    # ############### Plot the model ###############
    def plotFigure1(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.clear()

        self.correlation = self.df_data.corr()
        print(self.correlation)

        # Create a heatmap graph that will show which features of the model are correlated to each other and to what degree
        sns_plot = sns.heatmap(self.correlation, xticklabels=self.correlation.columns,
                               yticklabels=self.correlation.columns, cmap='RdBu')

        self.figure.add_subplot(sns_plot)
        self.figure.suptitle('Correlation Between Features')
        self.figure.tight_layout(pad=3)

        return self.canvas

    def plotFigure2(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        acc = self.history.history['mae']
        val_acc = self.history.history['val_mae']
        loss = self.history.history['loss']
        epochs = range(1, len(loss) + 1)

        ax.plot(epochs, loss, 'y', label='Training loss')
        ax.plot(epochs, val_acc, 'r', label='Validation loss')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        return self.canvas

    def plotFigure3(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.clear()

        y_pred_copy = self.y_predictions
        for i, val in enumerate(y_pred_copy):
            if int(val) == 0:
                y_pred_copy[i] = 1
            y_pred_copy[i] = int(val)
        pred_df = pd.DataFrame(y_pred_copy)
        actual_df = pd.DataFrame(self.y_test)
        labels = ['1', '2', '3']
        pred_count = pred_df.value_counts()
        actual_count = actual_df.value_counts()

        # the x label locations
        x = np.arange(len(labels))
        width = 0.35

        ax = self.figure.add_subplot(111)

        bars1 = ax.bar(x - width / 2, pred_count, width, label='Predictions')
        bars2 = ax.bar(x + width / 2, actual_count, width, label='Actual')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Number of Occurrences')
        ax.set_title('Occurrences by Predictions and Actual Outcomes')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(bars1, padding=3)
        ax.bar_label(bars2, padding=3)

        self.figure.tight_layout(pad=4)
        return self.canvas

    # Make a number of predictions based on the compiled model
    def predictionView(self):
        # tally up and print out whether or not the predictions were correct on main test data
        count = 0
        total = len(self.y_test)
        for i, val in enumerate(self.y_test):
            print('Predicted values are:', self.y_predictions[i], ' Real values are: ', val, ' Difference: ',
                  self.y_predictions[i] - val)
            y_pred_whole = int(self.y_predictions[i])
            diff = y_pred_whole - val

            if y_pred_whole != val:
                count += 1

        win_rate = 100 - ((count / total) * 100)
        result = 'Total miss:' + str(count) + ' Out of: ' + str(total) + ' So: ' + str(
            count / total) + ' Win rate: ' + str(win_rate)
        print(result)
        return result

        # Blind test results check
        print()
        print('Blind results:')
        self.y_blind_test_predictions = self.model.predict(self.x_blind_test_scaled)
        y_blind_test_prediction_not_scaled = self.model.predict(self.x_test)

        # tally up and print out whether or not the predictions were correct on blind test data
        test_count = 0
        test_total = len(self.y_blind_test)
        for i, val in enumerate(self.y_blind_test):
            print('Predicted values are:', self.y_blind_test_predictions[i], ' Real values are: ', val, ' Difference: ',
                  self.y_blind_test_predictions[i] - val)
            y_test_pred_whole = int(self.y_blind_test_predictions[i])
            diff_test = y_test_pred_whole - val
            #     if((test_y_prediction[i] - val) >= 1.0 or (test_y_prediction[i] - val) >= -1.0):
            if y_test_pred_whole != val:
                test_count += 1

        test_win_rate = 100 - ((test_count / test_total) * 100)
        print('Total missed:', test_count, ' Out of: ', test_total, ' So: ', test_count / test_total, ' Win rate:',
              test_win_rate)

    # Single Prediction using NN
    def single_prediction(self, txt):
        scaler = StandardScaler()
        scaler.fit(self.x_train)

        # Break the roulette wheel into 3 sections and group neighbor numbers together
        wheel_section_1 = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27]
        wheel_section_2 = [13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1]
        wheel_section_3 = [20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
        result = None
        try:
            list(txt.split(' '))
        except:
            # sys.exc_info()[0]
            return str('Follow Instructions')

        input_list = list(txt.split(","))
        print('Raw prediction input: ', input_list)
        try:
            list(map(int, input_list))
        except:
            # sys.exc_info()[0]
            return str('Follow instructions')

        input_list = list(map(int, input_list))

        # transform the ball position into section of the wheel value.
        if input_list[0] in wheel_section_1:
            input_list[0] = 1
        if input_list[0] in wheel_section_2:
            input_list[0] = 2
        if input_list[0] in wheel_section_3:
            input_list[0] = 3

        numb = self.model.predict(scaler.transform([input_list]))
        print('prediction transformed input: ', input_list, 'prediction output:', numb[0][0])
        if int(numb[0][0]) <= 1:
            result = wheel_section_1
        elif int(numb[0][0]) >= 3:
            result = wheel_section_3
        else:
            result = wheel_section_2
        return result

    # Get the mse and mae from the neural network
    def cal_mse_mae(self):
        self.mse_nn, self.mae_nn = self.model.evaluate(self.x_test_scaled, self.y_test)
        # mse_neural, mae_neural = model.evaluate(x_test, y_test)
        result = {'mse': self.mse_nn, 'mae': self.mae_nn}
        print('MSE from neural net: ', self.mse_nn)
        print('MAE from neural net: ', self.mae_nn)
        return result

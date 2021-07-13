import sys
import io
import pandas as pd
from Neural_Network import NN
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from main_stacked_window import Ui_MainWindow
from pandasModel import PandasModel


class MainWindow:
    def __init__(self):
        # Main Window variables init
        self.main_window = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        self.ui.stackedWidget.setCurrentWidget(self.ui.pg_logIn)

        self.passcode = ''
        self.single_prediction_input = ''
        self.summary = 'Overview of model performance: '
        self.pred_view = None

        # Get console error and output and store it into err and out
        self.out, self.err = io.StringIO(), io.StringIO()
        sys.stdout = self.out
        sys.stderr = self.err

        # page 1 set up action widgets
        self.ui.btn_LogIn.clicked.connect(self.show_page2)
        self.ui.le_passwordInput.textChanged[str].connect(self.update_login_te)

        # page 2 set up action widgets
        self.ui.btn_build_2.clicked.connect(self.show_page3)

        # page 3 set up action widgets
        self.ui.btn_makePred_2.clicked.connect(self.make_prediction)
        self.ui.le_predictionLe_2.textChanged[str].connect(self.update_prediction_input)
        self.ui.btn_toMaintView.clicked.connect(self.show_maintenance_page)

        # page 4 set up action widgets
        self.ui.btn_backToModel.clicked.connect(self.back_to_summary_page)

    # Show the main window
    def show(self):
        self.main_window.show()

    # Screen 2 setup and show
    def show_page2(self):
        # passcode input validation(0000)
        if self.login():
            self.ui.lb_errorLb.setText('')
            self.add_df_to_table_view()
            self.ui.stackedWidget.setCurrentWidget(self.ui.pg_dataView)
        else:
            self.ui.lb_errorLb.setText('The passcode you entered is not correct!')

    # Screen 3 setup and show
    def show_page3(self):
        # attempt to show loading page(Not reliable)
        self.show_loading_page()
        # Do data transformations on dataframe
        NN.dataTransform(NN)
        NN.defineXY(NN)
        # Normalize values by column
        NN.scaleValues(NN)
        NN.buildModel(NN)
        # Run predictions based on compiled model
        NN.prediction_test(NN)
        # Add plotted graphs to the window
        self.ui.hl_graphContainer.addWidget(NN.plotFigure1(NN))
        self.ui.hl_graphContainer.addWidget(NN.plotFigure2(NN))
        self.ui.hl_graphContainer.addWidget(NN.plotFigure3(NN))
        self.pred_view = NN.predictionView(NN)
        self.update_model_summary()
        self.ui.stackedWidget.setCurrentWidget(self.ui.pg_modelSummary)

    # Setup and show reporting page
    def show_maintenance_page(self):
        # walk through the predictions and label/print each prediction and actual outcome. Compute the difference
        for i, val in enumerate(NN.y_test):
            temp_str = 'Predicted values are: ' + str(NN.y_predictions[i]) + ' Real values are: ' + str(
                val) + ' Difference: ' + \
                       str(NN.y_predictions[i] - val)
            self.ui.tb_fullPredictions.append(temp_str)

        # Get errors and console output. Concat
        results = self.out.getvalue()
        errors = self.err.getvalue()
        full = errors + results

        self.ui.tb_dataView.setText(pd.DataFrame.to_string(NN.df_data))
        self.ui.tb_errorLogs.setText(full)
        self.ui.stackedWidget.setCurrentWidget(self.ui.pg_reporting)

    def back_to_summary_page(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.pg_modelSummary)

    def show_loading_page(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.pg_loading)

    def update_model_summary(self):
        mse_mae = NN.cal_mse_mae(NN)
        full_summary = self.summary + 'The means squared error(mse) from our model is : ' + str(mse_mae.get(
            'mse')) + ',' + 'The mean absolute error(mae) from our model is: ' + str(mse_mae.get(
            'mae')) + '.' + ' The closer to 0 our mse and mae is the better, This is way of gauging the accuracy of the model. For a clearer picture, if we used a dataset of 36 roulette rolls, this would be the outcome: ' + self.pred_view
        self.ui.lb_modelSummaryLb.setText(full_summary)

    # Keep track of input into prediction text field
    def update_prediction_input(self, text):
        self.single_prediction_input = text

    def make_prediction(self):
        pre = 'Numbers to bet on: '
        result = NN.single_prediction(NN, self.single_prediction_input)
        result_str = pre + str(result)
        self.ui.lb_predLb_2.setText(result_str)

    # Keep track of input into passcode text field
    def update_login_te(self, text):
        self.passcode = text

    def login(self):
        if self.passcode == '0000':
            auth = True
            self.login_suc()
            print('Passcode Entered: ', self.passcode, 'Authenticate? ', auth)
            return True
        return False

    # If log in is successful, load the dataframe from input data
    def login_suc(self):
        NN.loadData(NN)

    # Create a view from the dataframe modeler to display in table view correctly
    def add_df_to_table_view(self):
        model = PandasModel(NN.df_data)
        self.ui.tv_modelData_2.setModel(model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

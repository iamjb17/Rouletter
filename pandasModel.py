import sys
import typing

import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from Neural_Network import NN


# Create a view from the dataframe modeler to display in table view correctly
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data=pd.DataFrame()):
        QAbstractTableModel.__init__(self)
        print('in init')
        self.data = data.copy()

    def rowCount(self, parent=None):
        return self.data.shape[0]

    def columnCount(self, parent=None):
        return self.data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self.data.iloc[index.row(), index.column()])
            if role == Qt.EditRole:
                print('in edit', str(self.data.iloc[index.row(), index.column()]))
                return str(self.data.iloc[index.row(), index.column()])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid():
            print(len(value))
            row = index.row()
            col = index.column()
            value = str(value)
            value = value.strip()

            # While editing the table view, make sure that the input complies with the table data ant types
            if col == 0:
                if not value.isnumeric():
                    return False
                tmp_value = int(value)
                if tmp_value not in range(0, 36):
                    return False
                NN.df_data.iloc[row, col] = tmp_value
            if col == 1:
                if value != 'CCW' and value != 'CW':
                    return False
                NN.df_data.iloc[row, col] = value
            if col == 2:
                if value != 'CCW' and value != 'CW':
                    return False
                NN.df_data.iloc[row, col] = value
            if col == 4:
                if value != 'North' and value != 'West' and value != 'East' and value != 'South':
                    return False
                NN.df_data.iloc[row, col] = value
            if col == 5:
                if not value.isnumeric():
                    return False
                tmp_value = int(value)
                if tmp_value not in range(0, 9999):
                    return False
                NN.df_data.iloc[row, col] = tmp_value
            if col == 6:
                if not value.isnumeric():
                    return False
                tmp_value = int(value)
                if tmp_value not in range(0, 9999):
                    return False
                NN.df_data.iloc[row, col] = tmp_value
            if col == 7:
                if not value.isnumeric():
                    return False
                tmp_value = int(value)
                if tmp_value not in range(0, 9999):
                    return False
                NN.df_data.iloc[row, col] = tmp_value
            if col == 8:
                if not value.isnumeric():
                    return False
                tmp_value = int(value)
                if tmp_value not in range(0, 36):
                    return False
                NN.df_data.iloc[row, col] = tmp_value

            self.data.iloc[row, col] = value
            self.dataChanged.emit(index, index, (Qt.EditRole,))
            return True
        return False

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.data.columns[col]
        return None

    def flags(self, index):
        fl = super(self.__class__, self).flags(index)
        fl |= Qt.ItemIsEditable
        fl |= Qt.ItemIsSelectable
        fl |= Qt.ItemIsEnabled
        # fl |= Qt.ItemIsDragEnabled
        # fl |= Qt.ItemIsDropEnabled
        return fl


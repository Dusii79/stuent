from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QMainWindow, QMenuBar, QToolBar, QTextEdit, QAction, QApplication,
                             qApp, QMessageBox, QFileDialog, QLabel, QHBoxLayout, QGroupBox,
                             QComboBox, QGridLayout, QLineEdit, QSlider, QPushButton)
from PyQt5.QtGui import *
from PyQt5.QtGui import QPalette, QImage, QPixmap, QBrush
from PyQt5.QtCore import *
import sys
import cv2 as cv
import cv2
import numpy as np
import time
from pylab import *
import os
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class Window(QMainWindow):
    path = ' '
    change_path = "change.png"#被处理过的图像的路径
    IMG1 = ' '
    IMG2 = 'null'

    def __init__(self):
        super(Window, self).__init__()
        # 界面初始化

        self.createMenu()#创建左上角菜单栏
        self.cwd = os.getcwd()#当前工作目录
        self.image_show()
        self.label1 = QLabel(self)
        self.initUI()

    # 菜单栏
    def createMenu(self):
        # menubar = QMenuBar(self)
        menubar = self.menuBar()
        menu1 = menubar.addMenu("文件")
        menu1.addAction("打开")
        menu1.triggered[QAction].connect(self.menu1_process)

    #展示大图片
    def image_show(self):
        self.lbl = QLabel(self)
        self.lbl.setPixmap(QPixmap('source.png'))
        self.lbl.setAlignment(Qt.AlignCenter)  # 图像显示区，居中
        self.lbl.setGeometry(35, 35, 800, 700)
        self.lbl.setStyleSheet("border: 2px solid black")

    def initUI(self):
        self.setGeometry(50, 50, 900, 800)
        self.setWindowTitle('mnist识别系统')
        palette = QPalette()
        palette.setColor(self.backgroundRole(), QColor(255, 255, 255))
        self.setPalette(palette)

        self.label1.setText("TextLabel")
        self.label1.move(100,730)
        self.show()
    # 菜单1处理
    def menu1_process(self, q):
        self.path = QFileDialog.getOpenFileName(self, '打开文件', self.cwd,
                                                "All Files (*);;(*.bmp);;(*.tif);;(*.png);;(*.jpg)")
        self.image = cv.imread(self.path[0])
        self.lbl.setPixmap(QPixmap(self.path[0]))
        cv2.imwrite(self.change_path, self.image)
        transforms1 = transforms.Compose([
            transforms.ToTensor()
        ])
        self.label1.setText("识别中")
        img = Image.open(self.change_path)
        img = img.convert("L")
        img = img.resize((224, 224))
        tensor = transforms1(img)
        print(tensor.shape)
        tensor = tensor.type(torch.FloatTensor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
        tensor = tensor.reshape((1, 1, 224, 224))
        print(tensor.shape)
        y = net(tensor)
        print(y)
        print(torch.argmax(y))
        self.label1.setText(str(int(torch.argmax(y))))
if __name__ == '__main__':
    net = torch.load('cnn.pt')
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# GPU配置
print("检测GPU...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"找到 {len(gpus)} 个GPU:")
        for gpu in gpus:
            print(f" - {gpu.name}")
        
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已启用GPU内存动态增长")
            
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"已设置使用GPU: {gpus[0].name}")
            
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("GPU验证计算结果:", c)
                print("GPU成功配置完成!")
                
        except RuntimeError as e:
            print(f"设置GPU内存增长时出错: {e}")
    else:
        print("未检测到可用GPU，将使用CPU训练（这会明显降低训练速度）")
        print("如果您有兼容的GPU，请确保已安装GPU版本的TensorFlow和合适的CUDA/cuDNN")
        
except Exception as e:
    print(f"GPU检测过程中发生错误: {e}")
    print("将使用CPU继续运行")

print(f"TensorFlow版本: {tf.__version__}")
print(f"使用的设备: {tf.config.list_logical_devices()}")


class HandwritingInput:
    def __init__(self, parent, grid_size=20, cell_size=14):
        self.parent = parent
        self.grid_size = grid_size
        self.cell_size = cell_size

        # 初始化画布
        self.canvas = tk.Canvas(parent,
                              width=grid_size*cell_size,
                              height=grid_size*cell_size,
                              bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 绘图状态
        self.grid_state = np.zeros((grid_size, grid_size))
        self.drawing = False
        self.erasing = False
        self.history_stack = []
        self.redo_stack = []
        self.last_pos = None
        
        # 实时预测相关
        self.real_time_prediction = False
        self.prediction_delay = 100  # 降低延迟到100毫秒
        self.prediction_timer = None
        self.last_prediction_time = 0  # 添加最后预测时间记录

        # 加载模型和权重
        if os.path.exists('mnist_model.h5'):
            print("加载预训练MNIST模型...")
            self.cnn_model = load_model('mnist_model.h5')
            # 提取模型权重用于可视化
            self.extract_model_weights()
        else:
            print("未找到预训练模型，将使用随机权重进行可视化...")
            # 临时使用随机权重进行可视化
            input_nodes = 25
            hidden_nodes = 25
            output_nodes = 11
            self.weights = {
                'input_hidden': np.random.uniform(-1, 1, (input_nodes, hidden_nodes)),
                'hidden_output': np.random.uniform(-1, 1, (hidden_nodes, output_nodes))
            }

        # 绑定事件
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<Button-3>', lambda e: self.start_drawing(e, erase=True))
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<B3-Motion>', lambda e: self.draw(e, erase=True))
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        self.canvas.bind('<ButtonRelease-3>', self.stop_drawing)

        # 初始化网格
        self.draw_grid()

    def draw_grid(self):
        """绘制背景网格"""
        for i in range(self.grid_size):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.grid_size*self.cell_size, fill='#ddd')
            self.canvas.create_line(0, x, self.grid_size*self.cell_size, x, fill='#ddd')

    def start_drawing(self, event, erase=False):
        """开始绘制"""
        self.drawing = True
        self.erasing = erase
        self.history_stack.append(self.grid_state.copy())
        self.redo_stack.clear()
        self.last_pos = (event.x, event.y)
        self.draw(event, erase)

    def draw(self, event, erase=False):
        """持续绘制"""
        if not self.drawing:
            return

        # 计算网格坐标
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # 使用高斯分布模拟笔触
            self.apply_brush(x, y, erase)

            # 绘制线条
            if self.last_pos:
                self.canvas.create_line(
                    self.last_pos[0], self.last_pos[1],
                    event.x, event.y,
                    width=10, fill='black' if not erase else 'white'
                )
            self.last_pos = (event.x, event.y)

            # 实时预测 - 优化预测逻辑
            if self.real_time_prediction:
                current_time = time.time()
                if current_time - self.last_prediction_time > 0.1:  # 每100ms最多预测一次
                    if self.prediction_timer:
                        self.parent.after_cancel(self.prediction_timer)
                    self.prediction_timer = self.parent.after(10, self.predict_digit)
                    self.last_prediction_time = current_time

    def apply_brush(self, x, y, erase):
        """应用笔刷效果"""
        radius = 2  # 笔刷半径
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    distance = math.sqrt(dx**2 + dy**2)
                    intensity = max(0, 1 - distance/radius)

                    if erase:
                        self.grid_state[ny][nx] = 0.0
                    else:
                        self.grid_state[ny][nx] = min(1.0, self.grid_state[ny][nx] + intensity)

                    # 更新单元格颜色
                    gray = int(255 * (1 - self.grid_state[ny][nx]))
                    color = f'#{gray:02x}{gray:02x}{gray:02x}'
                    self.canvas.itemconfig(f'cell_{ny}_{nx}', fill=color)

    def stop_drawing(self, event):
        """停止绘制"""
        self.drawing = False
        self.last_pos = None

        # 实时预测 - 立即进行最后一次预测
        if self.real_time_prediction:
            if self.prediction_timer:
                self.parent.after_cancel(self.prediction_timer)
            self.predict_digit()

    def redraw_canvas(self):
        """重绘整个画布"""
        self.canvas.delete('all')
        self.draw_grid()
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                gray = int(255 * (1 - self.grid_state[y][x]))
                color = f'#{gray:02x}{gray:02x}{gray:02x}'
                self.canvas.create_rectangle(
                    x*self.cell_size, y*self.cell_size,
                    (x+1)*self.cell_size, (y+1)*self.cell_size,
                    fill=color, outline='', tags=f'cell_{y}_{x}'
                )

    def get_grid_state(self):
        """获取当前网格状态 - 28x28无需调整大小"""
        # 直接返回数组，添加批次和通道维度
        return self.grid_state

    def clear_canvas(self):
        """清除画布内容"""
        self.grid_state = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete('all')
        self.draw_grid()
        
    def line_color(self, weight):
        """根据权重值设置连接线颜色"""
        color = 'green' if weight > 0 else 'red'
        alpha = min(abs(weight), 1.0)
        return f'#{int(alpha*100):02x}ff{int(alpha*100):02x}' if color == 'green' else f'#ff{int(alpha*100):02x}{int(alpha*100):02x}'
    
    def draw_neural_network(self):
        """绘制神经网络可视化图"""
        self.nn_frame = ttk.LabelFrame(self.parent, text="神经网络可视化")
        self.nn_frame.grid(row=0, column=1, padx=10, pady=10)

        self.nn_width = 800
        self.nn_height = 700
        self.nn_canvas = tk.Canvas(self.nn_frame, width=self.nn_width, height=self.nn_height, bg='white')
        self.nn_canvas.grid(row=0, column=0, padx=5, pady=5)

        # 神经网络层的节点数
        input_nodes = 25
        hidden_nodes = 25
        output_nodes = 11

        # 计算每层的x坐标
        layer_x = [50, 400, 650]  # 输入层、隐藏层、输出层的x坐标

        # 节点半径
        node_radius = 10
        
        # 绘制输入层
        input_y_spacing = self.nn_height / (input_nodes + 1)
        hidden_y_spacing = self.nn_height / (hidden_nodes + 1)
        output_y_spacing = self.nn_height / (output_nodes + 1)
        
        for i in range(input_nodes):
            y = (i + 1) * input_y_spacing
            self.nn_canvas.create_oval(
                layer_x[0] - node_radius, y - node_radius,
                layer_x[0] + node_radius, y + node_radius,
                fill='white', outline='black', tags=f'input_{i}'
            )
            # 连接到隐藏层
            for j in range(hidden_nodes):
                hidden_y = (j + 1) * hidden_y_spacing
                self.nn_canvas.create_line(
                    layer_x[0] + node_radius, y,
                    layer_x[1] - node_radius, hidden_y,
                    fill=self.line_color(self.weights['input_hidden'][i][j]), width=0.5
                )

        # 绘制隐藏层
        for j in range(hidden_nodes):
            y = (j + 1) * hidden_y_spacing
            self.nn_canvas.create_oval(
                layer_x[1] - node_radius, y - node_radius,
                layer_x[1] + node_radius, y + node_radius,
                fill='white', outline='black', tags=f'hidden_{j}'
            )
            # 连接到输出层
            for k in range(output_nodes):
                output_y = (k + 1) * output_y_spacing
                self.nn_canvas.create_line(
                    layer_x[1] + node_radius, y,
                    layer_x[2] - node_radius, output_y,
                    fill=self.line_color(self.weights['hidden_output'][j][k]), width=0.5
                )

        # 绘制输出层
        for k in range(output_nodes):
            y = (k + 1) * output_y_spacing
            self.nn_canvas.create_oval(
                layer_x[2] - node_radius, y - node_radius,
                layer_x[2] + node_radius, y + node_radius,
                fill='white', outline='black', tags=f'output_{k}'
            )
            # 添加标签
            label_text = str(k) if k < 10 else "未识别"
            self.nn_canvas.create_text(
                layer_x[2] + 20, y,
                text=label_text,
                font=('Arial', 8)
            )

        # 添加层标签
        self.nn_canvas.create_text(layer_x[0], 20, text="输入层 (25)", font=('Arial', 10))
        self.nn_canvas.create_text(layer_x[1], 20, text="隐藏层 (25)", font=('Arial', 10))
        self.nn_canvas.create_text(layer_x[2], 20, text="输出层 (11)", font=('Arial', 10))

    def create_and_train_cnn_model(self):
        """创建并训练定制CNN模型"""
        # 构建卷积神经网络模型 - 与MNIST使用相同结构
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(11, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        # 加载训练数据
        if os.path.exists('train_data.npy') and os.path.exists('train_labels.npy'):
            train_data = np.load('train_data.npy')
            train_labels = np.load('train_labels.npy')
        else:
            print("未找到训练数据，请确保train_data.npy和train_labels.npy存在")
            return None

        # 训练模型
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        model.fit(train_data, train_labels, epochs=50, batch_size=32,
                validation_split=0.2, callbacks=callbacks)

        # 保存模型
        model.save('cnn_digit_model.h5')
        return model

    def load_and_train_with_mnist(self):
        """使用MNIST数据集预训练模型"""
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import train_test_split
        
        # 确认GPU使用情况
        print("\n检查当前设备:")
        devices = tf.config.list_logical_devices()
        is_using_gpu = any(d.device_type == "GPU" for d in devices)
        print("模型将在", "GPU" if is_using_gpu else "CPU", "上训练")
        
        # 加载MNIST数据集
        print("正在加载MNIST数据集...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 数据预处理
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # 添加通道维度
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # 添加未识别类别的样本（减少数量）
        noise_samples = np.random.normal(0.5, 0.2, (800, 28, 28, 1))
        noise_samples = np.clip(noise_samples, 0, 1)
        noise_labels = np.ones(800) * 10
        
        # 合并数据集
        x_all = np.vstack([x_train, noise_samples])
        y_all = np.concatenate([y_train, noise_labels])
        
        # 打乱数据
        indices = np.arange(len(x_all))
        np.random.shuffle(indices)
        x_all = x_all[indices]
        y_all = y_all[indices]
        
        # 分割训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(
            x_all, y_all,
            test_size=0.2,
            random_state=42,
            stratify=y_all  # 确保每个类别的比例一致
        )
        
        # 创建数据增强器
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # 验证集不需要数据增强
        val_datagen = ImageDataGenerator()
        
        # 构建改进的CNN模型
        model = Sequential([
            # 第一个卷积块
            Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # 第二个卷积块
            Conv2D(48, kernel_size=(3, 3), padding='same'),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # 全连接层
            Flatten(),
            Dense(96, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            Dropout(0.4),
            Dense(11, activation='softmax')
        ])
        
        # 使用Adam优化器，但降低学习率
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # 创建训练回调
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.00001
            ),
            ModelCheckpoint(
                'mnist_model_best.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # 训练参数
        epochs = 10  # 增加epochs但使用early stopping
        batch_size = 64
        
        # 计算steps
        steps_per_epoch = len(x_train) // batch_size
        validation_steps = len(x_val) // batch_size
        
        # 训练模型
        print("开始训练模型...")
        start_time = time.time()
        
        # 使用fit方法训练
        history = model.fit(
            train_datagen.flow(x_train, y_train, batch_size=batch_size),
            validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 打印训练历史
        print("\n训练历史:")
        for epoch in range(len(history.history['loss'])):
            print(f"Epoch {epoch+1}/{len(history.history['loss'])} - "
                  f"Train Loss: {history.history['loss'][epoch]:.4f}, "
                  f"Train Accuracy: {history.history['accuracy'][epoch]:.4f}, "
                  f"Validation Loss: {history.history['val_loss'][epoch]:.4f}, "
                  f"Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}")
        
        # 评估模型
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n测试集评估:")
        print(f"测试损失 (Loss): {test_loss:.4f}")
        print(f"测试准确率 (Accuracy): {test_acc*100:.2f}%")
        
        # 预测与真实值比较
        self.evaluate_model_predictions(model, x_test, y_test)
        
        # 保存模型
        model.save('mnist_model.h5')
        print("\n模型已保存为 'mnist_model.h5'")
        print(f"\n训练完成! 总耗时: {time.time() - start_time:.2f}秒")
        
        # 保存模型权重到类属性
        self.cnn_model = model
        return model

    def evaluate_model_predictions(self, model, x_test, y_test, samples=20):
        """评估模型预测并与真实值比较"""
        print("\n预测值与真实值比较 (随机样本):")
        
        # 随机选择样本
        indices = np.random.choice(len(x_test), samples, replace=False)
        x_samples = x_test[indices]
        y_true = y_test[indices]
        
        # 预测
        y_pred_prob = model.predict(x_samples, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # 打印结果
        print(f"{'样本':^6} | {'真实值':^6} | {'预测值':^6} | {'是否正确':^8} | {'置信度':^10}")
        print("-" * 50)
        
        correct = 0
        for i in range(samples):
            confidence = y_pred_prob[i, y_pred[i]] * 100
            is_correct = y_true[i] == y_pred[i]
            if is_correct:
                correct += 1
            
            true_label = int(y_true[i])
            true_text = str(true_label) if true_label < 10 else "未识别"
            pred_text = str(y_pred[i]) if y_pred[i] < 10 else "未识别"
            
            print(f"{i+1:^6} | {true_text:^6} | {pred_text:^6} | {'✓' if is_correct else '✗':^8} | {confidence:.2f}%")
        
        # 打印准确率
        accuracy = correct / samples * 100
        print(f"\n样本准确率: {accuracy:.2f}% ({correct}/{samples})")
        
        # 计算整个测试集的混淆矩阵
        y_pred_all = np.argmax(model.predict(x_test, verbose=0), axis=1)
        
        # 输出每个类别的准确率
        for cls in range(11):
            cls_indices = (y_test == cls)
            if np.sum(cls_indices) > 0:
                cls_acc = np.mean(y_pred_all[cls_indices] == cls) * 100
                cls_name = str(cls) if cls < 10 else "未识别"
                print(f"类别 {cls_name} 的准确率: {cls_acc:.2f}%")

    def predict_digit(self):
        """预测画布上的数字"""
        # 检查模型是否存在
        if not hasattr(self, 'cnn_model'):
            if os.path.exists('mnist_model.h5'):
                print("加载预训练MNIST模型...")
                self.cnn_model = load_model('mnist_model.h5')
            else:
                print("未找到预训练模型，开始训练MNIST模型...")
                self.load_and_train_with_mnist()

        # 获取当前绘制的图像并调整为28x28
        input_image = self.get_grid_state()
        
        # 确保它是标准化的0-1浮点数
        input_image = input_image.reshape(1, 28, 28, 1).astype('float32')
        
        # 预测
        predictions = self.cnn_model.predict(input_image, verbose=0)
        probabilities = predictions[0]
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit] * 100
        result_text = str(predicted_digit) if predicted_digit < 10 else "未识别"
        print(f"预测结果: {result_text} (置信度: {confidence:.2f}%)")
        
        # 可以添加一个标签来显示结果
        if hasattr(self, 'result_label'):
            self.result_label.config(text=f"预测: {result_text}\n置信度: {confidence:.2f}%")
        
        # 在神经网络可视化中高亮显示激活的节点
        self.update_neural_network_visualization(probabilities)
        
        return predicted_digit, confidence

    def update_neural_network_visualization(self, probabilities):
        """根据预测结果更新神经网络可视化"""
        if hasattr(self, 'nn_canvas'):
            # 重置所有输出节点的颜色
            for k in range(11):
                self.nn_canvas.itemconfig(f'output_{k}', fill='white')
                
            # 根据概率值设置输出节点的颜色
            for k in range(11):
                intensity = int(255 * probabilities[k])
                color = f'#{255-intensity:02x}ff{255-intensity:02x}'
                self.nn_canvas.itemconfig(f'output_{k}', fill=color)

    def toggle_real_time_prediction(self):
        """切换实时预测模式"""
        self.real_time_prediction = not self.real_time_prediction
        status = "开启" if self.real_time_prediction else "关闭"
        print(f"实时预测模式已{status}")
        if hasattr(self, 'real_time_button'):
            self.real_time_button.config(text=f"实时预测: {'开启' if self.real_time_prediction else '关闭'}")

    def extract_model_weights(self):
        """从CNN模型中提取权重用于可视化"""
        # 获取第一个全连接层的权重
        dense_layer = None
        for layer in self.cnn_model.layers:
            if isinstance(layer, Dense):
                dense_layer = layer
                break
        
        if dense_layer is not None:
            weights = dense_layer.get_weights()
            if len(weights) >= 2:  # weights[0]是权重矩阵，weights[1]是偏置
                # 简化权重以适应可视化
                input_size = weights[0].shape[0]
                output_size = weights[0].shape[1]
                
                # 将权重重塑为25x25和25x11的矩阵
                self.weights = {
                    'input_hidden': weights[0][:25, :25],  # 取前25个输入和隐藏节点
                    'hidden_output': weights[0][25:50, :11]  # 取接下来的25个隐藏节点到输出
                }


if __name__ == "__main__":
    root = tk.Tk()
    root.title("手写数字识别 - MNIST格式(28x28)")
    
    # 使用28x28像素的画布
    app = HandwritingInput(root, grid_size=28, cell_size=12)
    
    # 控制面板放在画布下方
    control_panel = ttk.Frame(root)
    control_panel.grid(row=1, column=0, padx=10, pady=10)
    
    # 清除按钮
    clear_button = ttk.Button(control_panel, text="清除画布", width=20, command=app.clear_canvas)
    clear_button.grid(row=0, column=0, padx=5, pady=5)
    
    # 预测按钮
    predict_button = ttk.Button(control_panel, text="预测数字", width=20, command=app.predict_digit)
    predict_button.grid(row=0, column=1, padx=5, pady=5)
    
    # 训练按钮
    train_button = ttk.Button(control_panel, text="使用MNIST训练", width=20, command=app.load_and_train_with_mnist)
    train_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    
    # 结果显示标签
    app.result_label = ttk.Label(control_panel, text="预测结果将显示在这里", font=('Arial', 14))
    app.result_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    # 实时预测按钮
    app.real_time_button = ttk.Button(control_panel, text="实时预测: 关闭", width=20, command=app.toggle_real_time_prediction)
    app.real_time_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    
    # 绘制神经网络可视化
    app.draw_neural_network()
    
    root.mainloop()    
import csv
import numpy as np
import matplotlib.pyplot as plt
import jieba
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TextVectorizer(object):
    def __init__(self):
        self.vectorizer = None
        self.label = None

    def _get_vocabulary(self, vector):
        """
        获取词汇表
        """
        return set([word for sublist in vector for word in sublist])

    def _create_bow_vector(self, words, vocab):
        """
        创建词袋向量
        """
        bow_vector = [0] * len(vocab)
        for word in words:
            if word in vocab:
                index = list(vocab).index(word)
                bow_vector[index] += 1
        return bow_vector

    def read_file(self, file_path, skip_header=True):
        """
        读取CSV文件的内容。
        参数:
            file_path (str): CSV文件的路径。
            skip_header (bool): 是否跳过表头数据，默认为True。
        返回:
            list: 包含CSV文件内容的列表。
        """
        print(f'{get_timestamp()} 读取原始数据集文件: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            if skip_header:
                # 跳过表头数据
                f.readline()
            reader = csv.reader(f)
            return [row for row in reader]  # 读取csv文件的内容

    def split_data(self, data):
        """
        将数据分割为标签和数据。
        参数:
            data (list): 数据行的列表，第一个元素是标签。
        返回:
            numpy.ndarray: 标签数组。
            numpy.ndarray: 连接元素后的数据数组。
        """
        label = [row[0] for row in data]
        data = [','.join(row[1:]) for row in data]

        # 转换为numpy数组
        n_label = np.array(label)
        n_data = np.array(data)

        return n_label, n_data

    def load_stop_words(self, file_path):
        """
        读取停用词文件。
        参数:
            file_path (str): 停用词文件的路径。
        返回:
            list: 包含停用词的列表。
        """
        print(f'{get_timestamp()} 读取停用词文件: {file_path}')
        # 从csv文件中读取停用词
        stopwords = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过第一行表头
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                stopwords.append(row[0])
        return stopwords

    def save_vector_data(self, label, vector, file_path_save):
        """
        处理数据的函数，包括读取文件、切分数据、向量化和保存向量化后的数据到新文件中。
        参数:
            label (numpy.ndarray): 标签数组。
            vector (list): 向量化数据。
            file_path_save (str): 向量化数据保存的路径。
        """

        print(f'{get_timestamp()} 向量化处理完毕，保存至: {file_path_save}')
        # 将向量化的数据保存到新文件中
        with open(file_path_save, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['label'] + ['x' + str(i) for i in range(len(vector[0]))])
            for i in range(len(label)):
                writer.writerow([label[i]] + vector[i])

    def text_to_vector(self, data, stopwords=None, use_jieba=True):
        print(f'{get_timestamp()} 开始数据集汉字的向量化处理')

        # 逐个汉字进行分词
        vector = []
        for review in data:
            words = []
            for word in review:
                if word not in stopwords:
                    words.append(word)
            vector.append(words)

        # 使用jieba分词
        if use_jieba:
            print(f'{get_timestamp()} 使用jieba库进行分词...')
            # jieba.enable_paddle()
            vector = [list(jieba.cut(''.join(words), use_paddle=True)) for words in vector]

        # 词袋模型的向量化
        vocab = self._get_vocabulary(vector)
        vectorized_data = [self._create_bow_vector(words, vocab) for words in vector]

        self.vectorizer = vectorized_data
        return vectorized_data


class MLModel(object):
    def __init__(self):
        self.clf = None
        self.accuracy = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name='Decision Tree'):
        print(f'{get_timestamp()}  开始训练{model_name}模型...')
        if model_name == 'Decision Tree':
            self.clf = DecisionTreeClassifier()
        elif model_name == 'KNN':
            self.clf = KNeighborsClassifier()
        elif model_name == 'Naive Bayes':
            self.clf = GaussianNB()
        elif model_name == 'SVM':
            self.clf = SVC()
        elif model_name == 'LogisticRegression':
            self.clf = LogisticRegression()
        elif model_name == 'RandomForestClassifier':
            self.clf = RandomForestClassifier()

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        self.accuracy = np.mean(y_pred == y_test)
        print(f'{get_timestamp()} {model_name}预测准确率：{self.accuracy}')

        return self.accuracy


class TextProcessingPipeline(object):
    def __init__(self):
        self.text_vectorizer = TextVectorizer()
        self.ml_model = MLModel()
        self.label = None
        self.data = None

    def process_data(self, file_path_read, file_path_save='vectorized_data.csv', use_jieba=True):

        print(f'{get_timestamp()} 读取原始数据集文件: {file_path_read}')
        # 读取原始数据集
        listdata = self.text_vectorizer.read_file(file_path_read)
        # 分割数据集
        label, data = self.text_vectorizer.split_data(listdata)
        # 读取停用词
        stopwords = self.text_vectorizer.load_stop_words('./stop_words.csv')
        # 根据传入的vec_type参数选择向量化方式
        vector = self.text_vectorizer.text_to_vector(data, stopwords, use_jieba)
        # 保存处理后的数据到本地
        self.text_vectorizer.save_vector_data(label, vector, file_path_save)
        # 读取保存的数据文件
        self.read_processed_data(file_path_save)

    def read_processed_data(self, file_path, skip_header=True):
        print(f'{get_timestamp()} 读取向量化数据集文件: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过第一行表头
            if skip_header:
                f.readline()
            reader = csv.reader(f)
            data = [row for row in reader]

        # 将标签转换为数值类型
        self.label = [int(row[0]) for row in data[1:]]

        # 将特征数据转换为数值类型
        self.data = [[float(x) for x in row[1:]] for row in data[1:]]

        return self.label, self.data

    def train_and_evaluate_models(self, use_jieba=True):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.2)
        model_names = ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM', 'LogisticRegression', 'RandomForestClassifier']
        accuracies = []

        for model_name in model_names:
            self.ml_model.train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
            accuracies.append(self.ml_model.accuracy)

        if use_jieba:
            print('使用jieba分词的情况:')
        else:
            print('不使用jieba分词的情况:')
        print(model_names)
        print(accuracies)

        return model_names, accuracies


def plot_accuracy_comparison(model_names, accuracies_jieba, accuracies_no_jieba):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35

    # 绘制第一组柱状图
    jieba_bars = ax.bar(x - width / 2, accuracies_jieba, width, label='with jieba')

    # 为第一组柱状图添加数据标签
    for i, v in enumerate(accuracies_jieba):
        ax.text(x[i] - width / 2, v, f"{v:.2f}", ha='center', va='bottom')

    # 绘制第二组柱状图
    no_jieba_bars = ax.bar(x + width / 2, accuracies_no_jieba, width, label='without jieba')

    # 为第二组柱状图添加数据标签
    for i, v in enumerate(accuracies_no_jieba):
        ax.text(x[i] + width / 2, v, f"{v:.2f}", ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Different Models Accuracy Comparison')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # 实例化处理流程对象并执行
    pipeline = TextProcessingPipeline()

    # 测试使用jieba进行分词的结果
    pipeline.process_data('./中文外卖评论数据集.csv', 'vectorized_data_nojieba.csv', use_jieba=True)
    model_names, accuracies_jieba = pipeline.train_and_evaluate_models(use_jieba=True)

    # 测试不使用jieba而只进行字分词结果
    pipeline.process_data('./中文外卖评论数据集.csv', 'vectorized_data_nojieba.csv', use_jieba=False)
    model_names, accuracies_no_jieba = pipeline.train_and_evaluate_models(use_jieba=False)

    # 绘制柱状图
    plot_accuracy_comparison(model_names, accuracies_jieba, accuracies_no_jieba)
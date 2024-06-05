import csv
import numpy as np


def read_file(file_path, skip_header=True):
    """
    读取CSV文件的内容。
    参数:
        file_path (str): CSV文件的路径。
        skip_header (bool): 是否跳过表头数据，默认为True。
    返回:
        list: 包含CSV文件内容的列表。
    """
    print(f'读取原始数据集文件: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as f:
        if skip_header:
            # 跳过表头数据
            f.readline()
        reader = csv.reader(f)
        return [row for row in reader]  # 读取csv文件的内容


def split_data(data):
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


def load_stop_words(file_path):
    """
    读取停用词文件。
    参数:
        file_path (str): 停用词文件的路径。
    返回:
        list: 包含停用词的列表。
    """
    print(f'读取停用词文件: {file_path}')
    # 从csv文件中读取停用词
    stopwords = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过第一行表头
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            stopwords.append(row[0])
    return stopwords


def text_to_vector(data, stopwords=None):
    """
    将数据中的汉字向量化。
    参数:
        data (numpy.ndarray): 包含汉字的数组。
        stopwords (list): 包含停用词的列表。
    返回:
        list: 使用词袋模型向量化的数据。
    """

    print('开始数据集汉字的向量化处理')
    # 逐个汉字进行分词
    vector = []
    for review in data:
        words = []
        for word in review:
            if word not in stopwords:
                words.append(word)
        vector.append(words)

    # 词袋模型的向量化
    # 举个例子，假设有以下数据:
    # 词汇表: {方, 便, 快, 捷, 味, 道, 可, 口, 递, 给, 力}
    # 初始向量: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 向量化过程:
    # 第一次: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # '方'出现1次
    # 第二次: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # '快'出现1次
    # 第三次: [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # '味'出现1次
    # 第四次: [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0]  # '可', '口'各出现1次

    # 第一步：构建词汇表
    vocab = set([word for sublist in vector for word in sublist])
    vectorized_data = []

    # 第二步：遍历字的内容
    for words in vector:
        # 第三步：生成一个长度为len(vocab)，内容都为0的向量
        bow_vector = [0] * len(vocab)
        # 第四步：找到词语在词汇表中的索引，然后在对应的位置上加1
        for word in words:
            if word in vocab:
                index = list(vocab).index(word)
                bow_vector[index] += 1
        vectorized_data.append(bow_vector)

    return vectorized_data


def process_data(label, vector, file_path_save):
    """
    处理数据的函数，包括读取文件、切分数据、向量化和保存向量化后的数据到新文件中。
    参数:
        label (numpy.ndarray): 标签数组。
        vector (list): 向量化数据。
        file_path_save (str): 向量化数据保存的路径。
    """

    print(f'向量化处理完毕，保存至: {file_path_save}')
    # 将向量化的数据保存到新文件中
    with open(file_path_save, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + ['x' + str(i) for i in range(len(vector[0]))])
        for i in range(len(label)):
            writer.writerow([label[i]] + vector[i])


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def decision_tree(train_data_file_path='vectorized_data.csv', test_size=0.2):
    """
    决策树模型的训练和评估。
    参数:
        train_data_file_path (str): 向量化数据的文件路径。
        test_size (float): 测试集的比例，默认为0.2。
    """
    print('开始加载训练数据...')
    # 读取文件
    with open(train_data_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # 数据切分
    label = [row[0] for row in data[1:]]
    vector = [row[1:] for row in data[1:]]

    # 训练集和测试集切分
    X_train, X_test, y_train, y_test = train_test_split(vector, label, test_size=0.2)

    print('开始训练决策树模型...')
    # 数据预测
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 评估
    print('开始决策树预测...')
    accuracy = np.mean(y_pred == y_test)
    print(f'预测准确率：{accuracy}')


if __name__ == "__main__":
    # 读取文件
    listdata = read_file('./中文外卖评论数据集.csv')
    # 对数据进行切分
    label, data = split_data(listdata)
    # 读取停用词
    stopwords = load_stop_words('./stop_words.csv')
    # 汉字数据进行向量化
    vector = text_to_vector(data, stopwords)
    # 处理数据
    process_data(label, vector, 'vectorized_data.csv')

    # 决策树模型的训练和评估
    decision_tree()

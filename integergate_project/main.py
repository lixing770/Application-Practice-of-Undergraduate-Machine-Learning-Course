import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据集加载
data = pd.read_csv('data.txt', sep='\t', header=None, names=['label', 'message'])



# 2. 数据集基本信息
print("前几行数据:")
print(data.head())

print("\n缺失值情况:")
print(data.isnull().sum())

print("\n类别分布:")
print(data['label'].value_counts())

# 3. 类别分布可视化
plt.figure(figsize=(6, 4))
sns.countplot(data['label'])
plt.title('Category Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Normal Message', 'Spam Message'])
plt.show()

# 4. 文本预处理
# 加载停用词
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])

# 分词和去停用词
def preprocess_text(text):
    words = jieba.cut(text)  # 使用jieba分词
    words = [word for word in words if word not in stopwords and len(word) > 1]  # 去除停用词和长度小于2的词
    return ' '.join(words)

data['processed_message'] = data['message'].apply(preprocess_text)

# 5. 数据集划分
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
print("\n训练集和测试集的大小:")
print("训练集:", train_data.shape)
print("测试集:", test_data.shape)

# 6. 文本向量化
# 使用TF-IDF向量化处理
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_data['processed_message'])
X_test = tfidf.transform(test_data['processed_message'])
y_train = train_data['label']
y_test = test_data['label']

print("\nTF-IDF 向量化后的特征矩阵维度:")
print("训练集:", X_train.shape)
print("测试集:", X_test.shape)

# 7. 词云分析
def generate_wordcloud(text, title, dpi=300):
    wordcloud = WordCloud(font_path='./simhei.ttf', background_color='white').generate(text)
    plt.figure(dpi=dpi)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

spam_text = ' '.join(train_data[train_data['label'] == 1]['processed_message'])
ham_text = ' '.join(train_data[train_data['label'] == 0]['processed_message'])
generate_wordcloud(spam_text, "Spam Message Wordcloud")
generate_wordcloud(ham_text, "Normal Message Wordcloud")

# 8. 模型训练与评估
# (1) 朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# (2) 逻辑回归
lr_model = LogisticRegression(max_iter=100, tol=1e-4, solver='lbfgs')
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# (3) 支持向量机
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 9. 模型评估
def evaluate_model(model_name, y_true, y_pred, dpi=300):
    print(f"{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print(f"{model_name} Confusion Matrix:\n", cm)

    # 绘制混淆矩阵热力图并设置 DPI
    plt.figure(figsize=(6, 4), dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal Message', 'Spam Message'],
                yticklabels=['Normal Message', 'Spam Message'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# 评估三个模型
evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Support Vector Machine", y_test, svm_pred)
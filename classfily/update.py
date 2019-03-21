import jieba
import re
from gensim.models.word2vec import Word2Vec


def clean_text(text):
    """
    采用结巴分词函数分词
    :param corpus: 待分词的Series序列
    :return: 分词结果，list
    """
    # 去除无用字符
    pattern = re.compile(r'[\sA-Za-z～()（）【】%*#+-\.\\\/:=：__,，。、;；“”""''’‘？?！!<《》>^&{}|=……]')
    corpus_ = text.apply(lambda s: re.sub(pattern, '', s))
    # 分词
    updatetext = corpus_.apply(jieba.lcut)
    # 过滤通用词
    updatetext = updatetext.apply(lambda cut_words: [word for word in cut_words if word not in self.stopword])
    return updatetext


def update_model(text):
    model = Word2Vec.load("dyword.model")  # 加载旧模型
    model.build_vocab(text, update=True)  # 更新词汇表
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
    return model

def main():
    update_text = ''
    jieba.load_userdict("dy_word.utf8")
    text = clean_text(update_text)## 分词
    model = update_model(text)
    # 保存模型
    model.save("dyword.model")


if __name__ == '__main__':
    main()
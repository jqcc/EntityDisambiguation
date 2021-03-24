import argparse
import numpy as np
import jieba
import jieba.posseg as pseg
import jieba.analyse as anse
import logging

from urllib import request
from lxml import etree

parser = argparse.ArgumentParser()
parser.add_argument('--embed_file', default=None, help='加载自定义词向量字典')
parser.add_argument('--embed_size', default=300, help="词向量大小, 默认300")
parser.add_argument('--discard', action='store_false', help="生成消歧句子向量时, 保留消歧词, 默认不保留")
opt = parser.parse_args()


class MultiEntityExtract(object):
    """docstring for MultiEntityExtract"""
    def __init__(self, config):
        super(MultiEntityExtract, self).__init__()
        self.opt = config
        self.embed_dict = self.load_embedding_dict()

    # 加载embedding字典
    def load_embedding_dict(self):
        embed_file_path = self.opt.embed_file
        if embed_file_path is None:
            embed_file_path = 'word_vec_300.bin'
        try:
            embed_dict = {}
            with open(embed_file_path, 'r', encoding='utf-8') as f:
                counter = 0
                for line in f: 
                    line = line.strip().split(' ')
                    if len(line) < self.opt.embed_size:
                        continue
                    else:
                        word = line[0]
                        vec = np.array([float(i) for i in line[1:]])
                        embed_dict[word] = vec
                        counter += 1

                    if counter % 10000 == 0:
                        print("%d word loaded" % counter)

                embed_dict['<pad>'] = np.zeros(self.opt.embed_size)
                print("loaded %d words totally" % counter)
        except Exception as e:
            logging.exception(e)
            raise ValueError('缺少embedding文件')

        return embed_dict

    # 根据单词获取对应的embedding
    def get_word_embedding(self, word):
        return self.embed_dict.get(word, self.embed_dict['<pad>'])
        
    # 对于给定句子提取出20个关键词计算平均作为句子的表示
    def get_sent_embedding(self, sentence):
        # 使用textrank算法提取出20个关键词 第四个参数用于过滤
        key_words = anse.extract_tags(sentence, topK=20)
        sent_embed = np.zeros(self.opt.embed_size)
        cnt = 0
        for word in key_words:
            if word in self.embed_dict:
                word_vec = self.get_word_embedding(word)
                sent_embed += word_vec
                cnt += 1

        return sent_embed if cnt == 0 else sent_embed / cnt

    # 计算输入句子的表示 opt.discard决定是否保留消歧词 默认不保留
    def get_input_embedding(self, word, sentence):
        if self.opt.discard:
            sentence = sentence.replace(word, "").strip()

        return self.get_sent_embedding(sentence)

    # 使用爬虫获取页面信息
    def get_html(self, url):
        return request.urlopen(url).read().decode('utf-8').replace('&nbsp', '')

    # 传入url获取页面中的语义项列表
    def get_entity_tags(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        # 取出多义项的名称及对应的页面链接
        items = selector.xpath('//li[@class="list-dot list-dot-paddingleft"]/div/a/text()')
        links = selector.xpath('//li[@class="list-dot list-dot-paddingleft"]/div/a/@href')

        # 组织成(tag_name, tag_link)数据项对
        entity_tags = []
        for i, l in zip(items, links):
            # 分词并分析对应的词性(分词, 词性)
            # 注: 可分析多义项页面 一般最后一个词为有意义的消歧概念
            tag = pseg.lcut(i)[-1]

            tlink = 'https://baike.baidu.com' + l
            entity_tags.append([tag.word, tlink])

        return entity_tags

    # 根据tag数据项对生成tag的语义向量
    def get_entity_vecs(self, tags):
        ret_vec = []
        for tag in tags:
            tag_name, tag_link = tag[0], tag[1]
            html = self.get_html(tag_link)
            selector = etree.HTML(html)
            meta_desc = selector.xpath('//meta[@name="description"]/@content')
            meta_kword = selector.xpath('//meta[@name="keywords"]/@content')
            context = "".join(meta_desc + [' '] + meta_kword)
            # print(context)

            ret_vec.append([tag_name, self.get_sent_embedding(context)])
                
        return ret_vec

    # 计算两个向量的余弦相似度
    def cosine_similarity(self, vec1, vec2):
        cos1 = np.sum(vec1 * vec2)
        cos21 = np.sqrt(sum(vec1 ** 2))
        cos22 = np.sqrt(sum(vec2 ** 2))
        similarity = cos1 / float(cos21 * cos22)
        if str(similarity) == 'nan':
            similarity = 0.0
        
        return similarity

    # 获取候选实体列表
    def get_candidate_entity(self, word):
        e_url = "https://baike.baidu.com/item/{}?force=1".format(request.quote(word))
        tags = self.get_entity_tags(e_url)
        return self.get_entity_vecs(tags)

    def detect_word(self, word, sentence):
        # 对句子生成一个语义表示
        sent_embed = self.get_input_embedding(word, sentence)
        # print(sent_embed)
        # 获取消歧词备选实体表是表示列表
        candidates = self.get_candidate_entity(word)
        # 基于相似度筛选出消歧后的实体概念
        ret = []
        for tag in candidates:
            ret.append([tag[0], self.cosine_similarity(sent_embed, tag[1])])

        candidates = sorted(ret, key=lambda x: x[1], reverse=True)
        return candidates[:3]


def main():
    mee = MultiEntityExtract(opt)
    while 1:
        sentence = input("请输入一个句子: ").strip()
        word = input("请输入要消歧的单词: ").strip()

        if word not in jieba.lcut(word):
            print("该词似乎不是一个有效的词, 确认要继续么?(可能会消歧失败)")
            ans = input("请输入[y/n]")
            if ans != "y":
                continue
        
        ret = mee.detect_word(word, sentence)
        print("消歧结果: \n", ret)


if __name__ == '__main__':
    main()

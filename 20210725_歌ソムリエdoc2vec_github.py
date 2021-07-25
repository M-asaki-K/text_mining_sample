# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:35:23 2021

@author: uni21
"""

# import module
import pandas as pd
import glob
import os
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

"""
まずは歌詞データベースを呼び出します。
今回はアーティスト別（アーティストランキング上位100グループ、下記リンクより）に全歌詞を収集したcsvファイルを
hogeフォルダに作成しましたので、これを一つのデータベースに整形します。
https://ranking.net/rankings/best-music-artists
"""
# ディレクトリ変更
os.chdir('hoge')

# 空のDataFrameを定義
df = pd.DataFrame()

# .csvを含むファイルをpd.read_csv()で読み込む
for i in glob.glob("*.csv*"):
    tmp_df = pd.read_csv(i)
# DataFrameを連結する
    df = pd.concat([df, tmp_df])

df = df.reset_index()
df = df.dropna(subset = ["Lyric"])

"""
ここからは、歌詞データベースおよそ19000件をDoc2Vecによりベクトル化し、ロッキンジャパン2021の中止に対する渋谷陽一氏のコメントと
類似したものをランキング化します。そこで、データベースの0行目に渋谷陽一氏のコメントを挿入します。
http://rijfes.jp/2021/info/1928/
"""
objective = ["現在、たくさんの夏フェスが計画されています。全ての成功を祈っています。ですから、私たちの開催中止が他の夏フェスへの悪い影響を生まなければいいと切に願っています。その為にも最後まで開催を目指して頑張りたかったですが、ここで断念せざるを得ませんでした。悔しいですし、申し訳なく思います。音楽を止めるな、フェスを止めるな、という思いでたくさんの仲間が頑張っています。それは参加者の皆さんも一緒です。その強い思いが春のジャパン・ジャムの成功を支えたのだと思います。ジャパン・ジャムの会場でたくさんの方から「夏は絶対に開催してくださいね」と声をかけていただきました。「大丈夫、絶対に開催するから」と答えました。百パーセント、そう思っていました。ジャパン・ジャムの成功が夏開催への道筋をひらくものと信じていたので本当に残念です。無念です。フェス開催1カ月前という、ほぼスキーム変更が困難なタイミングでの要請であった為に、私たちにできることはほとんどありませんでした。政府のガイドライン、茨城県やひたちなか市による協力要請を遵守し、会場や県、市の皆さんと密な協議を重ねて開催の承認をいただいてきたのですが、医師会の方の危機感はそれを超えて大きく重かったということで、しっかり受け止めさせていただくしかありません。要望書が提出された翌週の7月5日(月)、茨城県医師会のホームページに、提出時の写真と要望書の内容がアップされていました。なぜか数時間で消えていましたが、多くの方が情報共有できるように再掲載いただけたらと思います。「ロック・イン・ジャパン　2021」の開催は中止になってしまいましたが、各地で夏フェスは開催されます。絶対に成功してほしいです。音楽を止めない、フェスを止めない、という思いは多くの音楽ファンが持つ共通の思いです。コロナ禍にあって障害はありますが、あの祝祭空間を私たちは守っていかなければなりません。そんな思いでこの文章も書かせていただいています。いつも最前線で戦い、状況を切り開く覚悟で頑張ってきましたが、今回は思いはかないませんでした。「ロック・イン・ジャパン　2021」にむかうアーティストの気持ち、フェスを楽しみにしている参加者の思い、それを考えると何とも言えない気持ちになります。悔しいです。申し訳ありません。でも私たちはこれからも最高のフェスを作り続けます。今回のことで、その思いはより強くなりました。繰り返しになりますが、これから開催される夏フェスの成功を強く願い、応援したいと思います。"]
train_data = df["Lyric"].tolist()
train_data = objective + train_data
train_data = train_data
train_data_df = pd.DataFrame(train_data)
train_data_sample = train_data_df.iloc[0:10, :]

"""
今回取り扱う歌詞は日本語ですので、そのままだと単語ごとの分かち書きが出来ていません。
これを解決するために、Mecabパッケージを導入し文章を分かち書きに自動整形します。
"""
def wakati(target_text):
    t = MeCab.Tagger("-Owakati")
    result = t.parse(target_text)
    return result

train_data_df_wakati = train_data_df[0].apply(lambda x: wakati(x))
train_data_df_wakati = pd.DataFrame(train_data_df_wakati)

"""
Doc2Vecを適用するため、各歌詞（渋谷氏コメント含む）を単語ごとにリスト化します。
"""
sentences = []
for text in train_data_df_wakati[0]:
    text_list = text.split(' ')
    sentences.append(text_list)

"""
Doc2Vecモデルにより、各歌詞に対するニューラルネットワークを学習し、ハイパーパラメータにてベクトル化を行います。
よく見ると色々調整すべきパラメータがあるので、まだ詰め代はあるかもしれません（ここは使用目的によりけりです。今回はアート作品なので、私が見て面白かったからよし、としています）。
"""
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
model = Doc2Vec(documents, vector_size=2, window=5, min_count=1, workers=4)

"""
0行目に格納されたテキスト、即ち渋谷氏のコメントに対し、最も似ているものの順に歌詞をランキングします。
デフォルトでは上位10件が表示されますが、もっと見たければ上位n件を指定することも可能です。
most_similar(文書名, topn=n)
"""
res = model.docvecs.most_similar(0)
res_df = pd.DataFrame(res)

results_top = pd.DataFrame(res_df[0].apply(lambda x: train_data[x]))
results_top.to_csv("hoge.csv", encoding = "cp932")

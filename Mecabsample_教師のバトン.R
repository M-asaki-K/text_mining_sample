library(tidyverse)
library(rtweet)
library(RMeCab)
library(twitteR)
library(wordcloud2)
library(dplyr)

tweets <- search_tweets("#教師のバトン", n = 1000, include_rts = FALSE)
tweets %>% 
  select(text)
tweets$created_at
View(tweets$text)

## Reading 単語感情極性表(Semantic Orientations of Words)
sowdic <- read.table("http://www.lr.pi.titech.ac.jp/~takamura/pubs/pn_ja.dic", sep=":", 
                     col.names=c("term","kana","pos","value"), colClasses=c("character","character","factor","numeric"), 
                     fileEncoding="Shift_JIS")

# 名詞＋品詞で複数の候補がある場合は最大値を採用します
sowdic2 <- aggregate(value ~ term + pos, sowdic, max)

# write.table for check frequency
write.table(tweets$text,"d.txt")

# Frequency
frq_Zimin <- RMeCabFreq("d.txt")


# wordcloud
frq_Zimin_exp <- frq_Zimin %>%
  filter(Info1 %in% c("名詞","形容詞","動詞")) %>% 
  filter(Freq > 15) %>% 
  filter(!(Info2 %in% c("数", "サ変接続", "一般", "非自立")))  %>% 
  filter(!(Term %in% c("する", "いる", "ある", "てる","なる","れる","やる", "これ", "たち", "できる", "良", "あれ","それ","思う","せる", "内田", "省","https","られる","的","私","人","そう","見る","言う","何","者","目","方", "会","年","さ","いう")))

frq_Zimin_exp

wordcloud2(frq_Zimin_exp[, c("Term", "Freq")])

# ポジティブとネガティブの感情分析
frq_Zimin_posinega <- frq_Zimin %>%
  rename(term = Term) %>% 
  inner_join(sowdic2, by = "term") %>% 
  #0以上をポジティブと定義
  mutate(posi_nega = if_else(value > 0, "Posi", "Nega"))

# 円グラフ用に集計
posinega_sum <- frq_Zimin_posinega %>% 
  group_by(posi_nega) %>% 
  summarize(n())

# 円グラフを表示
pie(posinega_sum$`n()`, labels = posinega_sum$posi_nega, col = rainbow(2), clockwise = TRUE)

用 Latex 撰寫之論文：[LaTex_new.pdf](https://github.com/yy1200/FinRL_TS/files/12726143/LaTex_new.pdf)

# FinRL_TS
## 研究主題及背景描述
從歷史資料中找出與現今最相似的環境以進行訓練，目的為使投資組合的報酬最大。

投資組合管理是市場上投資公司的一項重要交易任務。主要是將資金依照權重做適當的分配，在每段時間點上將資金投資到不同的資產類別上，藉由分散投資來達到分散風險以及優化投資組合報酬的目的。

## 文獻回顧及missing
### 傳統的金融資產配置模型
* The Markowitz Optimization Enigma: Is ‘Optimized’ Optimal?
Markowitz提出的現代投資組合理論或均值方差理論選擇給定風險水平下的最大預期收益，以增加實現預期收益的確定性。但其有一些侷限性，因為不容易確定未來的資產波動率，且模型太過簡單，無法學習複雜的市場環境。

過往將機器學習、深度學習運用在投資組合上的例子與問題探討：
### ML / DL
* Machine Learning and Portfolio Optimization
採用正則化和交叉驗證這兩種機器學習方法，用於投資組合優化，並藉由約束投資組合風險和收益的樣本方差，降低估計誤差。
* Deep Learning for Portfolio Optimization
提出了一個框架，利用深度學習模型直接優化投資組合的Sharpe ratio，省去傳統的預測步驟，並允許通過更新模型參數來優化投資組合權重。
* Deep Learning in Finance
使用深度學習分層模型解決了金融預測和分類問題，並試圖預測價格走向或趨勢。
但機器學習無法考慮每個交易之間的關係，會產生滑價、手續費等問題，所以目前的研究大多使用 RL 來處理連續交易。

### RL
* Adversarial Deep Reinforcement Learning in Portfolio Management
使用DDPG、DPG和PPO進行投資組合優化，並採用對抗性方法訓練模型，顯著提升了回測期間的日均收益率和夏普比率，最後發現DPG算法具有更好的性能。
* Cryptocurrency Portfolio Management with Deep Reinforcement Learning
使用 CNN 和 DRL 來分配和調整加密貨幣的權重。
* Deep Reinforcement Learning for Portfolio Management
在投資組合管理中使用DRL並增加做空機制和設計套利機制來優化投資決策，可以獲得超額收益並以極低的交易成本率進行交易。
* A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
將深度強化學習應用於投資組合管理和數字貨幣資產的分配，然後通過加權優化投資組合，並使用連續加權來解決因離散動作引起的問題。
然而，這些算法在測試階段仍然表現出較弱的泛化能力，並存在概念漂移。

### 概念漂移是動態環境中數據分佈隨時間序列變化的現象
* Combining Online Classification Approaches for Changing Environments
提出兩種可以用來處理概念漂移的方法，第一個方法是先從歷史數據中訓練agent，然後用新的可用數據不斷更新agent；第二種方法是使用變化檢測，當變化量超過閾值時就重新訓練agent。
* Financial portfolio optimization with online deep reinforcement learning and restricted stacked autoencoder—DeepBreath
採用第一種方法適應特徵漂移，因為市場上很容易出現與過去相似的交易趨勢。而第二種方法因不使用歷史數據，故沒有這種先驗知識。
如果概念漂移不斷發生，agent傾向於採取短期投資策略，這將不利於長期投資。所以概念漂移是我們需要解決的問題之一。

### 此外，DQN 算法訓練單個任務需要很長時間。
* Policy Distillation
提出了政策蒸餾的概念，為一個師生框架，用於將教師模型中學到的知識轉移到學生模型中，且可以在不降低性能的情況下壓縮網絡規模。
* Universal Trading for Order Execution with Oracle Policy Distillation
提出了解決不完善環境的策略蒸餾方法。這種方法是一個師生學習框架，在這個框架中，擁有完美信息的老師被訓練成神諭來找出最優的交易策略，學生通過模仿老師的最優行為模式來學習。
然而，以上所有方法都是學生向老師學習，沒有交互更新。但是，我們的老師會根據學生的反饋進行更新，通過選擇合適的環境來幫助學生解決概念漂移的問題。

## Proposed method
在金融市場中，股票價格波動受到投資人買賣活動的影響。然而，影響金融市場股價和投資者決策的因素有很多，包括機構交易行為、公司最新消息和財務狀況等，因此很難觀察到所有這些因素及其相應的市場狀況直接​​地。因此，本實驗選擇價格趨勢的變化來研究，決策過程可以用部分可觀測馬可夫決策過程（POMDP）來描述。

該研究將每天將資金重新分配到不同的資產，以分散風險並最大化投資回報。

![圖片2](https://user-images.githubusercontent.com/92247082/232473912-f1643330-9646-4f64-afe9-be1a7db7742f.png)

在這個實驗中，我們使用歷史股票價格作為訓練資料來訓練學生，使其有足夠的知識來做出投資組合管理決策。然而，訓練資料中存在概念漂移，因此我們嘗試額外訓練一名教師，其目的是選擇與今天最相似的環境。然後，老師會在需要時為學生提供最合適的訓練資料。因此，可以大幅減少概念漂移。

POMDP $(S,A,R,P,gamma)$ 元組集用來說明我們的架構：

State：比較所有目前和過去的環境。計算前一日（t-1）、週（t-5）、月（t-20）、季度（t-60）、半年（t-125）和年度（t-250）的總漲跌幅)與每日(t) 收盤價，並將它們進行比較以找到最相似的環境。

Action(a)：狀態計算出的兩個機率分別是過去環境機率和當前環境機率。如果前者大於後者，則該環境將包含在模型訓練資料集中。透過這種方法，將其分為0,1類型，0不包含在訓練集中，否則為1。

Reward：使用action得到的訓練集進行訓練，計算累計收益 { $r_1, r_2,...,r_n$ } 的平均R。

### Problem formulation
用MSE方法找到最相似的環境進行訓練，並將actor保存為學生的預訓練模型。

**Student**

學生透過老師篩選出的100個環境進行訓練，並以獲得的獎勵來更新老師。

**Teacher**

教師的目的是選擇100個與今天最相似的環境，並透過與環境互動並不斷更新來學習學生訓練的最優策略。

## 實驗：

**投資組合資產和時間**

共有 3 組投資組合，分別為：
1. 市值前十大 [‘AAPL’,‘MSFT’,‘AMZN’,‘TSLA’,‘GOOGL’,‘NVDA’,‘BRK-A’,‘META’,‘UNH’,'JNJ ']
2. 產業龍頭 [‘XOM’,‘BHP’,‘UPS’,‘AMZN’,‘WMT’,‘JNJ’,‘BRK-A’,‘AAPL’,‘GOOGL’,‘NEE’]
3. 成長股 [‘TSLA’,‘MELI’,‘NFLX’,‘AMZN’,‘META’,‘CRM’,‘GOOGL’,‘VRTX’,‘TMUS’,‘HAL’]

以下實驗的 TRAIN_START_DATE (歷史起始日期)均設定為 2012-06-01

我們使用 YahooFinance 下載資料集，提取股票的每日收盤價，並將其轉換為百分比變化。此外，我們也設計了四個版本：
1. No Sample -- 根據過去八年中每天收盤價的歷史數據進行訓練
2. Random -- 以過去8年為歷史數據，隨機選出100個不同的環境進行訓練
3. MSE -- 從歷史中找與test前一天相似的環境，並將test的前約 1、5、20、60、125、250 天算成6 * 10(檔股票) 個值與歷史做比對算出MSE，以找出最相似的 100 個環境後的 125 天來做訓練。
![圖片1](https://user-images.githubusercontent.com/92247082/232473826-67ea46f3-04a7-4c62-9627-2393b24d7372.png)
4. Proposed Model -- 以過去8年為歷史數據，MSE選擇與測試期間前一天最相似的200個環境。然後，根據老師（T）給出的機率決定最適合的100個環境。取每個環境後的125天訓練學生（S），保存獲得的回報，每收集10個值更新一次T。


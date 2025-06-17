1.架構設計與動機
首先Backbone我使用基於 MobileNet V3-Small 的單一 Head 多任務學習架構，可同時處理語意分割 (Segmentation)、目標偵測 (Detection) 與影像分類 (Classification)。架構核心包括：

使用參數量僅約 2.5M 的 MobileNet V3-Small 作為 Backbone，兼具效能與輕量化。

透過一層 1×1 卷積層 (Neck) 將特徵通道壓縮至 160，降低後續運算負擔。

在 Unified Head 中以兩層卷積同時輸出三項任務所需的張量，確保持續共享特徵。

採用 Elastic Weight Consolidation (EWC) 在第二與第三階段引入正則化項，抑制跨任務遺忘。

1.1 Backbone：MobileNet V3-Small

首先選擇 MobileNet V3-Small 作為特徵提取主幹，原因在於其參數量少、推論速度快，並內建 SE (Squeeze-and-Excitation) 與 Hard-Swish 激活，可在低計算成本下保有良好表徵能力（Howard et al., ICCV 2019）。此外，該網路在多個層級輸出特徵，利於多任務同時從高層語義與低層細節中受益。

1.2 Neck：通道壓縮與映射

在 backbone 輸出特徵後緊接一層 1×1 卷積，將通道數從 576 壓縮至 160，再經 Batch Normalization 與 ReLU。如此可維持特徵分佈穩定並提供非線性，同時顯著減少 Unified Head 的運算量。

1.3 Unified Head：雙層卷積結構

Unified Head 由兩層卷積組成，第一層為 3×3 卷積 (160→160) 後接 ReLU，用以融合空間資訊；第二層為 1×1 卷積，將通道映射為 Detection、Segmentation 和 Classification 三項任務所需的輸出張量。輸出張量會依類別與任務切分：Detection 部分包含 (cx, cy, w, h, conf) 及類別 logits；Segmentation 部分輸出 21 個類別 mask logits，後續上採樣至原始解析度；Classification 部分經 Global Average Pooling 得到最終 logits。

1.4 損失函數與 EWC 正則化

對於 Segmentation，採用 CrossEntropy Loss；對於 Detection，conf 分支在前 10 個 epoch 以 BCEWithLogitsLoss 作 warm-up，之後改用 Sigmoid Focal Loss (γ=1)；bbox 分支採 Smooth L1 Loss；classification 分支同樣採 CrossEntropy Loss。為抑制任務間遺忘，於第二階段 (Detection) 加入以第一階段 (Segmentation) Fisher Information 計算的 EWC 正則化項；於第三階段 (Classification) 則同時加入第一與第二階段的 EWC 項，分別以不同 λ 權重調整。

2. 訓練排程與防遺忘策略

我們將訓練劃分為三個階段：

第一階段 (Segmentation)：在 PASCAL VOC mini_seg 資料上訓練 10 個 epoch，學習語意分割任務，記錄 mIoU_base，並計算該階段參數的重要度。

第二階段 (Detection)：在 mini_COCO_det 資料上訓練 40 個 epoch，同時加入第一階段的 EWC 正則化，抑制對分割關鍵參數的更新，以維持原先語意分割性能。

第三階段 (Classification)：在 mini_Imagenette_160 資料上訓練 10 個 epoch，採雙重 EWC 正則化 (第一階段 + 第二階段)，兼顧前兩任務的重要參數，並最終獲得分類性能。

3.訓練資料提取
我根據三個資料集中官方提供的標註檔，利用python程式對每個類別進行隨機抽取，並依照抽取的類別，從驗證集中找出對應類別的驗證照片

4.訓練結果

語意分割 mIoU 從基準 0.954 輕微提升至 0.975；

目標偵測 mAP@0.5 從基準 0.0015 幾乎歸零至 0.000；

影像分類 Top-1 準確率從基準 0.800 大幅下降至 0.533；

兩項未達到 ≤5% 性能跌幅標準。可能原因如下:

Dataset 與標註對應錯誤：

COCO 資料集中類別數達 63，但 cat2idx 或 grid 映射有誤，導致偵測 target 全為空或全部歸類為背景。

偵測資料讀取時 bbox 座標歸一化或像素轉換錯位，造成模型無法學習有效位置。

Grid 解析度不足：

構造 16×16 的單格格網對於 COCO 小物體無法精確定位，且類別過多時資訊密度過大，conf 輸出分散，導致 focal loss 無法收斂。

EWC 超參數設定不當：

Stage2 的 λ 值過大，導致偵測階段無法有效更新 backbone 與 neck 參數；Stage3 雙重 EWC 更進一步抑制分類學習。

Unified Head 結構瓶頸：

單一 Head 兩層卷積可能無法同時兼顧三項互異任務的專屬特徵需求，尤其偵測與分類對空間與全球特徵需求相悖。

Loss 權重不平衡：

各任務 loss 權重皆相同，但任務間梯度量級差異巨大，導致高 magnitude 的 segmentation loss 學習主導，其他任務梯度被抑制。

改進方向

檢查並修正偵測資料處理流程：

驗證 cat2idx 生成是否正確、bbox 坐標轉換是否與模型輸入維度對齊；

確認訓練與驗證資料讀取無誤，例如 mask 與分類 label 對應正確。

提高偵測解析度與多尺度特徵：

將 grid 大小改為 32×32 或加入輕量 FPN，提升空間分辨率；

採用 Adaptive Anchor 或 Anchor-free 方案降低對 grid 的依賴。

重新調整 EWC 超參數：

降低 Stage2 λ 或僅對部分重要參數施加正則化；

在偵測階段採用漸進式 EWC，前期弱化正則化後逐步增加強度。

分離 Head 結構或引入 Task-Specific Module：

針對 detect、class 兩項任務額外增設輕量分支，避免梯度互相衝突；

探索動態路徑選擇模組 (Dynamic Task Routing)。

Loss 權重與優化器配置微調：

依據梯度量級動態調整各任務 loss 權重 (GradNorm，PCGrad 方法)；

嘗試不同 learning rate 或 optimizer 參數，例如對 detect 分支使用更高 LR。

Knowledge Distillation 與 Multi-Task 梯度處理：

以大型 teacher 模型分別對 segmentation、detection、classification 進行蒸餾指導；

使用 gradient surgery 方法 (PCGrad、GradVac) 減少任務間梯度衝突。

5. 資源與效能
模型參數量（Parameters）:1.27M
訓練時間:36分鐘
推論速度::5.84ms

6.結論
本次實驗中，儘管語意分割性能略有提升，但目標偵測與影像分類均未達作業要求，主要因資料映射錯誤、單一 Head 結構與 EWC 設定過度限制等原因。未來可依上述方向逐步排查與優化，以期在維持輕量化與高效能的前提下，同時完成多任務學習目標。



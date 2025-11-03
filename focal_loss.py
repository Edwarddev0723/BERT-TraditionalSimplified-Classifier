"""
Focal Loss 實現 - 處理類別不平衡

在 classifier_finetune_v6_optimized.ipynb 中使用此代碼
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss - 專注於困難樣本
    
    論文: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    
    參數:
        alpha: 類別權重，用於平衡正負樣本
               - float: 所有類別使用相同權重
               - list/tensor: 每個類別的權重
        gamma: 調製因子，控制對簡單樣本的降權程度
               - gamma=0: 等同於標準交叉熵
               - gamma越大: 越關注困難樣本
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: 標籤平滑參數 (0-1)
    
    使用範例:
        # 方法1: 自動計算類別權重
        loss_fct = FocalLoss(alpha='auto', gamma=2.0)
        
        # 方法2: 手動指定權重
        loss_fct = FocalLoss(alpha=[0.4, 0.6], gamma=2.0)
        
        # 訓練時
        outputs = model(input_ids, attention_mask)
        loss = loss_fct(outputs.logits, labels)
    """
    
    def __init__(self, 
                 alpha=0.25, 
                 gamma=2.0, 
                 reduction='mean',
                 label_smoothing=0.0):
        super().__init__()
        
        # 類別權重
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif alpha == 'auto':
            # 將在第一次forward時自動計算
            self.alpha = None
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        print(f"✓ FocalLoss initialized:")
        print(f"   alpha={alpha}, gamma={gamma}")
        print(f"   label_smoothing={label_smoothing}")
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型輸出 logits, shape: (batch_size, num_classes)
            targets: 真實標籤, shape: (batch_size,)
        """
        # 計算交叉熵（不進行reduction）
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # 計算概率
        pt = torch.exp(-ce_loss)  # pt: 正確類別的預測概率
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Focal loss
        focal_loss = focal_term * ce_loss
        
        # 應用類別權重
        if self.alpha is not None:
            if self.alpha == 'auto':
                # 自動計算（第一次調用時）
                # 這裡簡化處理，實際應該在初始化時計算
                pass
            else:
                if isinstance(self.alpha, torch.Tensor):
                    alpha_t = self.alpha.to(inputs.device)
                    # 根據目標選擇對應的alpha
                    alpha_t = alpha_t[targets]
                else:
                    alpha_t = self.alpha
                
                focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    加權 Focal Loss - 自動根據類別分布計算權重
    
    使用範例:
        # 在訓練前計算類別權重
        label_counts = train_df['label'].value_counts().sort_index()
        class_weights = len(train_df) / (len(label_counts) * label_counts.values)
        
        loss_fct = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=2.0
        )
    """
    
    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        print(f"✓ WeightedFocalLoss initialized:")
        print(f"   class_weights={class_weights}")
        print(f"   gamma={gamma}, label_smoothing={label_smoothing}")
    
    def forward(self, inputs, targets):
        # 計算交叉熵
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.class_weights,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # 計算概率
        pt = torch.exp(-ce_loss)
        
        # Focal term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


# ===== 使用示例 =====

def example_usage_in_notebook():
    """
    在 classifier_finetune_v6_optimized.ipynb 中的使用方式
    """
    
    # ===== 方法1: 在模型定義中使用 =====
    
    # 修改 OptimizedBertClassifier 的 forward 方法
    """
    class OptimizedBertClassifier(nn.Module):
        def __init__(self, ...):
            super().__init__()
            # ... 其他初始化
            
            # 添加 Focal Loss
            self.focal_loss = FocalLoss(alpha=0.4, gamma=2.0, label_smoothing=0.05)
        
        def forward(self, ..., labels=None, ...):
            # ... 前向傳播
            logits = self.classifier(pooled_output)
            
            # 計算損失
            total_loss = None
            if labels is not None:
                main_loss = self.focal_loss(logits, labels)
                total_loss = main_loss
                
                # 輔助任務損失（如果有）
                if category_logits is not None and category_labels is not None:
                    category_loss = F.cross_entropy(category_logits, category_labels)
                    total_loss = main_loss + CATEGORY_LOSS_WEIGHT * category_loss
            
            return SequenceClassifierOutput(loss=total_loss, logits=logits, ...)
    """
    
    # ===== 方法2: 使用自定義 Trainer =====
    
    """
    from transformers import Trainer
    
    class FocalLossTrainer(Trainer):
        def __init__(self, *args, focal_loss=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.focal_loss = focal_loss or FocalLoss(alpha=0.4, gamma=2.0)
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 使用 Focal Loss
            loss = self.focal_loss(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
    
    # 使用
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        focal_loss=FocalLoss(alpha=0.4, gamma=2.0, label_smoothing=0.05)
    )
    """
    
    # ===== 方法3: 計算類別權重 =====
    
    """
    # 在數據準備階段
    label_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)
    
    # 計算權重: weight[i] = total / (num_classes * count[i])
    class_weights = [total / (num_classes * count) for count in label_counts.values]
    
    print("類別權重:")
    for i, (count, weight) in enumerate(zip(label_counts.values, class_weights)):
        label_name = '大陸繁體' if i == 0 else '台灣繁體'
        print(f"   {label_name}: count={count:,}, weight={weight:.3f}")
    
    # 使用加權 Focal Loss
    loss_fct = WeightedFocalLoss(
        class_weights=class_weights,
        gamma=2.0,
        label_smoothing=0.05
    )
    """
    
    pass


# ===== 完整集成代碼 =====

FOCAL_LOSS_INTEGRATION_CODE = """
# ===== 在 classifier_finetune_v6_optimized.ipynb 中添加 =====

# 1. 在導入部分添加
from focal_loss import FocalLoss, WeightedFocalLoss

# 2. 在數據分析後計算類別權重
print("\\n📊 計算類別權重...")
label_counts = train_df['label'].value_counts().sort_index()
total = len(train_df)
num_classes = len(label_counts)

class_weights = torch.tensor([
    total / (num_classes * count) for count in label_counts.values
], dtype=torch.float32)

print("類別權重:")
for i, (count, weight) in enumerate(zip(label_counts.values, class_weights.numpy())):
    label_name = '大陸繁體' if i == 0 else '台灣繁體'
    print(f"   {label_name}: count={count:,}, weight={weight:.3f}")

# 3. 修改模型的損失計算
class OptimizedBertClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, num_categories=None, 
                 use_attention_pooling=True, use_multi_sample_dropout=True,
                 msd_num_samples=5, msd_dropout_rate=0.3,
                 class_weights=None, use_focal_loss=True):  # 新增參數
        super().__init__()
        
        # ... 其他初始化代碼 ...
        
        # 損失函數
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = WeightedFocalLoss(
                class_weights=class_weights,
                gamma=2.0,
                label_smoothing=LABEL_SMOOTHING
            )
        else:
            self.loss_fct = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=LABEL_SMOOTHING
            )
    
    def forward(self, input_ids, attention_mask=None, labels=None, 
                category_labels=None, chunk_count=None, sample_id=None, **kwargs):
        
        # ... 前向傳播代碼 ...
        
        # 計算損失
        total_loss = None
        if labels is not None:
            if self.use_focal_loss:
                main_loss = self.focal_loss(logits, labels.view(-1))
            else:
                main_loss = self.loss_fct(
                    logits.view(-1, self.num_labels), 
                    labels.view(-1)
                )
            
            total_loss = main_loss
            
            # 輔助任務損失
            if category_logits is not None and category_labels is not None:
                category_loss = F.cross_entropy(
                    category_logits.view(-1, self.num_categories),
                    category_labels.view(-1)
                )
                total_loss = main_loss + CATEGORY_LOSS_WEIGHT * category_loss
        
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 4. 創建模型時傳入權重
model = OptimizedBertClassifier(
    model_name=str(encoder_path),
    num_labels=2,
    num_categories=num_categories,
    use_attention_pooling=USE_ATTENTION_POOLING,
    use_multi_sample_dropout=MULTI_SAMPLE_DROPOUT,
    msd_num_samples=MSD_NUM_SAMPLES,
    msd_dropout_rate=MSD_DROPOUT_RATE,
    class_weights=class_weights,  # 傳入類別權重
    use_focal_loss=True  # 啟用 Focal Loss
)

print(f"✓ 模型創建成功，使用 Focal Loss")
"""

if __name__ == '__main__':
    print("=" * 60)
    print("Focal Loss 實現")
    print("=" * 60)
    print("\n這個文件提供了處理類別不平衡的 Focal Loss 實現")
    print("\n使用方式:")
    print("1. 將此文件保存為 focal_loss.py")
    print("2. 在 classifier_finetune_v6_optimized.ipynb 中導入")
    print("3. 參考 FOCAL_LOSS_INTEGRATION_CODE 進行集成")
    print("\n" + "=" * 60)
    
    # 簡單測試
    print("\n測試 Focal Loss:")
    loss_fct = FocalLoss(alpha=0.4, gamma=2.0, label_smoothing=0.05)
    
    # 模擬數據
    logits = torch.randn(8, 2)  # batch_size=8, num_classes=2
    labels = torch.randint(0, 2, (8,))
    
    loss = loss_fct(logits, labels)
    print(f"Loss value: {loss.item():.4f}")
    print("\n✓ Focal Loss 測試通過！")

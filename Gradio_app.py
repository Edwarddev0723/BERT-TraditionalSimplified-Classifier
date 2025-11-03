"""
Gradio App for BERT Traditional Chinese Classifier
用於測試繁體中文分類模型（大陸繁體 vs 台灣繁體）的互動式介面
"""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# ========== 配置 ==========
REPO_ID = "renhehuang/bert-traditional-chinese-classifier"
LABELS = {0: "大陸繁體", 1: "台灣繁體"}
MAX_LEN, STRIDE = 384, 128

# ========== 初始化模型 ==========
print("🔄 載入模型與 tokenizer...")
device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=".cache")
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, cache_dir=".cache")
model.to(device).eval()
print(f"✅ 模型已載入至 {device}")

# ========== 工具函數 ==========
def chunk_encode(text, max_len=MAX_LEN, stride=STRIDE):
    """長文本分塊編碼"""
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) <= max_len - 2:
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        return_attention_mask=True, return_tensors="pt")
        return [enc]
    enc = tokenizer(text, truncation=True, max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_attention_mask=True,
                    return_tensors="pt")
    return [{"input_ids": enc["input_ids"][i:i+1],
             "attention_mask": enc["attention_mask"][i:i+1]}
            for i in range(len(enc["input_ids"]))]


@torch.inference_mode()
def predict_single(text: str):
    """單次推論"""
    if not text or not text.strip():
        return "⚠️ 請輸入文本", "", {}
    
    chunks = chunk_encode(text)
    probs_all = []
    for ch in chunks:
        logits = model(
            input_ids=ch["input_ids"].to(device),
            attention_mask=ch["attention_mask"].to(device)
        ).logits
        probs_all.append(F.softmax(logits, dim=-1).cpu())
    
    avg = torch.cat(probs_all, 0).mean(0)
    label_id = int(avg.argmax())
    confidence = float(avg[label_id])
    
    # 格式化輸出
    result_text = f"🏷️ **{LABELS[label_id]}**"
    confidence_text = f"📊 信心度: **{confidence:.2%}**"
    probabilities = {
        "大陸繁體": float(avg[0]),
        "台灣繁體": float(avg[1])
    }
    
    return result_text, confidence_text, probabilities


@torch.inference_mode()
def predict_voting(text: str, n_runs: int = 3):
    """多次投票推論（MC Dropout）"""
    if not text or not text.strip():
        return "⚠️ 請輸入文本", "", {}, ""
    
    chunks = chunk_encode(text)
    prev_training = model.training
    run_prob_list = []
    
    try:
        model.train()  # 啟用 dropout
        for _ in range(n_runs):
            probs_all = []
            for ch in chunks:
                logits = model(
                    input_ids=ch["input_ids"].to(device),
                    attention_mask=ch["attention_mask"].to(device)
                ).logits
                probs_all.append(F.softmax(logits, dim=-1).cpu())
            run_prob_list.append(torch.cat(probs_all, 0).mean(0))
    finally:
        model.train() if prev_training else model.eval()
    
    probs_stack = torch.stack(run_prob_list, 0)
    per_run_ids = probs_stack.argmax(-1).tolist()
    vote_counts = Counter(per_run_ids)
    mean_probs = probs_stack.mean(0)
    
    voted_id = max(vote_counts.items(), key=lambda kv: (kv[1], mean_probs[kv[0]].item()))[0]
    confidence = float(mean_probs[voted_id])
    
    # 格式化輸出
    result_text = f"🏷️ **{LABELS[voted_id]}**"
    confidence_text = f"📊 平均信心度: **{confidence:.2%}**"
    probabilities = {
        "大陸繁體": float(mean_probs[0]),
        "台灣繁體": float(mean_probs[1])
    }
    vote_info = f"🗳️ 投票結果: {vote_counts[voted_id]}/{n_runs} 次"
    
    return result_text, confidence_text, probabilities, vote_info


# ========== Gradio 介面 ==========
examples = [
    ["這個軟件的界面設計得很好。"],
    ["這個軟體的介面設計得很好。"],
    ["我需要下載這個程序到計算機上。"],
    ["我需要下載這個程式到電腦上。"],
    ["請打開視頻觀看教程。"],
    ["請打開影片觀看教學。"],
]

with gr.Blocks(title="BERT 繁體中文分類器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 BERT 繁體中文分類器
    
    區分「大陸繁體」與「台灣繁體」的 BERT 分類模型
    
    - 支援長文本自動分塊處理（max_len=384）
    - 提供單次推論與多次投票（MC Dropout）模式
    """)
    
    with gr.Tab("📝 單次推論"):
        with gr.Row():
            with gr.Column():
                input_single = gr.Textbox(
                    label="輸入文本",
                    placeholder="請輸入繁體中文文本...",
                    lines=5
                )
                btn_single = gr.Button("🚀 開始分類", variant="primary")
            
            with gr.Column():
                output_label_single = gr.Markdown(label="預測結果")
                output_conf_single = gr.Markdown(label="信心度")
                output_probs_single = gr.Label(label="機率分布", num_top_classes=2)
        
        gr.Examples(
            examples=examples,
            inputs=input_single,
            label="範例文本"
        )
        
        btn_single.click(
            fn=predict_single,
            inputs=input_single,
            outputs=[output_label_single, output_conf_single, output_probs_single]
        )
    
    with gr.Tab("🗳️ 投票推論（MC Dropout）"):
        with gr.Row():
            with gr.Column():
                input_voting = gr.Textbox(
                    label="輸入文本",
                    placeholder="請輸入繁體中文文本...",
                    lines=5
                )
                n_runs = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=3,
                    step=1,
                    label="投票次數",
                    info="推論次數越多，結果越穩定但速度較慢"
                )
                btn_voting = gr.Button("🚀 開始投票分類", variant="primary")
            
            with gr.Column():
                output_label_voting = gr.Markdown(label="預測結果")
                output_conf_voting = gr.Markdown(label="平均信心度")
                output_probs_voting = gr.Label(label="平均機率分布", num_top_classes=2)
                output_vote_info = gr.Markdown(label="投票統計")
        
        gr.Examples(
            examples=examples,
            inputs=input_voting,
            label="範例文本"
        )
        
        btn_voting.click(
            fn=predict_voting,
            inputs=[input_voting, n_runs],
            outputs=[output_label_voting, output_conf_voting, output_probs_voting, output_vote_info]
        )
    
    with gr.Tab("ℹ️ 關於模型"):
        gr.Markdown("""
        ## 模型資訊
        
        - **模型**: ckiplab/bert-base-chinese
        - **任務**: 繁體中文文本分類（大陸繁體 vs 台灣繁體）
        - **準確率**: 87.71%
        - **訓練樣本**: 156,824
        
        ## 標籤定義
        
        - **大陸繁體（中國繁體）**: 使用「软件、视频、程序、计算机」等詞彙
        - **台灣繁體**: 使用「軟體、影片、程式、電腦」等詞彙
        
        ## 功能特色
        
        - ✅ 長文本自動分塊處理（384 tokens，stride 128）
        - ✅ Focal Loss 處理類別不平衡
        - ✅ Multi-Sample Dropout 提升泛化
        - ✅ MC Dropout 投票提升穩健性
        
        ## 使用建議
        
        - 對於重要決策，建議使用「投票推論」模式並設定 5-10 次投票
        - 信心度 ≥ 85% 的預測較為可靠
        - 混用詞彙、專業術語或極短文本可能影響準確度
        
        ---
        
        📦 **模型倉庫**: [renhehuang/bert-traditional-chinese-classifier](https://huggingface.co/renhehuang/bert-traditional-chinese-classifier)
        
        📄 **授權**: Apache 2.0
        """)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

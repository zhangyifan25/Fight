import torch
import clip
import os

def generate_text_prototype(output_path, device='cuda'):
    """生成文本原型并对齐到256维"""
    print("正在加载CLIP模型...")
    
    # 加载CLIP（指定精度为float32）
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # 背景提示词
    text_prompts = [
        "a photo of ground", 
        "background", 
        "land", 
        "soil", 
        "terrain",
        "empty ground",
        "bare earth",
        "ground surface",
        "dirt",
        "grassland"
    ]
    
    print(f"使用提示词: {text_prompts}")
    
    # 将文本转为token
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        # 提取文本特征 [N, 512]
        text_features = clip_model.encode_text(text_tokens)
        
        # L2归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 取平均得到 [1, 512]
        text_prototype_512 = text_features.mean(dim=0, keepdim=True)
        
        print(f"文本原型形状 (512维): {text_prototype_512.shape}")
        print(f"数据类型: {text_prototype_512.dtype}")
        
        # ===== 修复：统一数据类型 =====
        # 将text_prototype_512转为float32
        text_prototype_512 = text_prototype_512.float()
        print(f"转换后数据类型: {text_prototype_512.dtype}")
        
        # 创建投影矩阵（使用float32）
        projection = torch.randn(512, 256).to(device)
        projection = projection / projection.norm(dim=0, keepdim=True)
        # projection已经是float32
        
        # 投影到256维
        text_prototype_256 = torch.mm(text_prototype_512, projection)  # [1, 256]
        text_prototype_256 = text_prototype_256 / text_prototype_256.norm(dim=-1, keepdim=True)
        
        print(f"文本原型形状 (256维): {text_prototype_256.shape}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存256维版本（保存为float32）
    torch.save(text_prototype_256.cpu(), output_path)
    print(f"文本原型已保存到: {output_path}")
    
    # 验证保存的文件
    loaded = torch.load(output_path)
    print(f"验证加载成功，形状: {loaded.shape}, 数据类型: {loaded.dtype}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    output_path = "./background_prototypes/text_prototype.pth"
    generate_text_prototype(output_path, device)
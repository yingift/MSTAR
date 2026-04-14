import torch

# --------------------- 你只需要改这里 ---------------------
p1 = "/data/liangyin/.cache/torch/hub/checkpoints/hug_cross_800.pth"    # 补充来源
p2 = "mstar_weights/mstar.pth"     # 主权重（保持不变）
save_path = "mstar_weights/mstar1.pth"  # 输出文件名
# --------------------------------------------------------

# 加载权重
print("加载 c1 (补充来源)...")
c1 = torch.load(p1, map_location='cpu')

print("加载 c2 (主权重)...")
c2 = torch.load(p2, map_location='cpu')

# 取出 model 部分
model1 = c1["model"]
model2 = c2["model"]

# 统计
added_keys = []
total1 = len(model1.keys())
total2 = len(model2.keys())

print(f"\nc1 模型 key 数量: {total1}")
print(f"c2 模型 key 数量: {total2}")
print("开始合并：c2 不变，只补充 c1 有但 c2 没有的 key...\n")

# 核心合并逻辑
for key in model1:
    if key not in model2:
        model2[key] = model1[key]
        added_keys.append(key)

# 打印新增了哪些 key
print(f"✅ 成功补充 {len(added_keys)} 个 key：")
for k in added_keys:
    print("   +", k)

# 保存合并后的权重（完整结构：model/optimizer/config/scaler/epoch）
c2["model"] = model2
torch.save(c2, save_path)

print(f"\n🎉 合并完成！已保存到：{save_path}")
print(f"最终模型 key 数量：{len(model2.keys())}")
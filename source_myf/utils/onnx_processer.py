import onnxruntime as ort
import numpy as np
import time
import onnx
from onnx import helper, compose, TensorProto
from onnx import numpy_helper

def set_ir_version(model, ir_version):
    model.ir_version = ir_version
    return model
import onnx

def convert_opset(model, target_opset):
    return onnx.version_converter.convert_version(
        model,
        target_opset
    )
def dump_outputs(path):
    m = onnx.load(path)
    print("Model:", path)
    for o in m.graph.output:
        shape = [
            d.dim_value if d.HasField("dim_value") else "?"
            for d in o.type.tensor_type.shape.dim
        ]
        print(" output:", o.name, shape)

dump_outputs("/home/roborock/docker_images_v1.8.x/docker_bushu/resnet18_-1.onnx")
dump_outputs("/home/roborock/docker_images_v1.8.x/docker_bushu/pointnet17.onnx")
# -------------------
# 模拟输入
# -------------------
image = np.random.rand(1,3,360,640).astype(np.float32)
points = np.random.rand(1,3,1024).astype(np.float32)

# -------------------
# 合并前推理
# -------------------
resnet = onnx.load("/home/roborock/docker_images_v1.8.x/docker_bushu/resnet18_-1.onnx")
graph = resnet.graph

# 1. 新增 Flatten 节点
flatten_node = helper.make_node(
    "Flatten",
    inputs=[graph.output[0].name],  # GAP 输出名字
    outputs=["resnet_flat"],        # 新输出
    axis=1
)
graph.node.append(flatten_node)

# 2. 修改模型输出为 Flatten 输出
graph.output[0].name = "resnet_flat"


# 3. 保存修改后的模型
onnx.save(resnet, "resnet18_flatten.onnx")
resnet = onnx.load("/home/roborock/IsaacLab/resnet18_flatten.onnx")
pointnet = onnx.load("/home/roborock/docker_images_v1.8.x/docker_bushu/pointnet17.onnx")
ppo = onnx.load("/home/roborock/IsaacLab/logs/rsl_rl/coarse_arm_lift/red_car/exported2/red_car.onnx")
ppo = onnx.version_converter.convert_version(
    ppo,
    17
)

TARGET_IR = min(
    resnet.ir_version,
    pointnet.ir_version,
    ppo.ir_version
)

resnet = set_ir_version(resnet, TARGET_IR)
pointnet = set_ir_version(pointnet, TARGET_IR)
ppo = set_ir_version(ppo, TARGET_IR)
resnet = compose.add_prefix(resnet, prefix="resnet_")
pointnet = compose.add_prefix(pointnet, prefix="pointnet_")
ppo = compose.add_prefix(ppo, prefix="ppo_")

sess_resnet = ort.InferenceSession("/home/roborock/IsaacLab/resnet18_flatten.onnx")
sess_pointnet = ort.InferenceSession("/home/roborock/docker_images_v1.8.x/docker_bushu/pointnet17.onnx")
sess_ppo = ort.InferenceSession("/home/roborock/IsaacLab/logs/rsl_rl/coarse_arm_lift/red_car/exported2/red_car.onnx")
# 先合并两个特征提取器
feat_model = compose.merge_models(resnet, pointnet, io_map=[])
graph = feat_model.graph

# L2 norm 节点
norm_img = helper.make_node(
    "LpNormalization",
    inputs=[graph.output[0].name],  # img_feat
    outputs=["img_feat_norm"],
    axis=1, p=2
)
norm_pc = helper.make_node(
    "LpNormalization",
    inputs=[graph.output[1].name],  # pc_feat
    outputs=["pc_feat_norm"],
    axis=1, p=2
)

# concat 节点
concat = helper.make_node(
    "Concat",
    inputs=["img_feat_norm", "pc_feat_norm"],
    outputs=["fused_feat"],
    axis=1
)

graph.node.extend([norm_img, norm_pc, concat])

# 更新 graph 输出
while len(graph.output) > 0:
    graph.output.pop()

graph.output.append(
    helper.make_tensor_value_info("fused_feat", TensorProto.FLOAT, None)
)
final_model = compose.merge_models(
    feat_model,
    ppo,
    io_map=[("fused_feat", ppo.graph.input[0].name)]
)
graph = final_model.graph

# 原 PPO 输出名
ppo_out = graph.output[0].name

# 常量 0.1
scale_const = numpy_helper.from_array(
    np.array(0.1, dtype=np.float32),
    name="scale_0p1"
)
graph.initializer.append(scale_const)

# Mul 节点
scale_node = helper.make_node(
    "Mul",
    inputs=[ppo_out, "scale_0p1"],
    outputs=["scaled_action"],
    name="ScaleAction"
)

graph.node.append(scale_node)

# 替换 graph 输出
graph.output[0].name = "scaled_action"
onnx.save(final_model, "yonghan.onnx")


# 输入：image + points
# 输出：img_feat, pc_feat

# warm-up
for _ in range(5):
    feat_img = sess_resnet.run(None, {"input": image})[0]
    print(feat_img.shape)
    feat_pc = sess_pointnet.run(None, {"points": points})[0]
    fused = np.concatenate([feat_img/np.linalg.norm(feat_img,axis=1,keepdims=True),
                            feat_pc/np.linalg.norm(feat_pc,axis=1,keepdims=True)], axis=1)
    action_value = sess_ppo.run(None, {"obs": fused})[0]

# 测耗时
t0 = time.time()
feat_img = sess_resnet.run(None, {"input": image})[0]
feat_pc = sess_pointnet.run(None, {"points": points})[0]
fused = np.concatenate([feat_img/np.linalg.norm(feat_img,axis=1,keepdims=True),
                        feat_pc/np.linalg.norm(feat_pc,axis=1,keepdims=True)], axis=1)
action_value = sess_ppo.run(None, {"obs": fused})[0]
t1 = time.time()
print("合并前推理耗时: {:.2f} ms".format((t1-t0)*1000))

# -------------------
# 合并后推理
# -------------------
sess_final = ort.InferenceSession("yonghan.onnx")

# warm-up
for _ in range(5):
    outputs = sess_final.run(
    None,
    {
        "resnet_input": image.astype(np.float32),
        "pointnet_points": points.astype(np.float32)
    }
)

# 测耗时
t0 = time.time()
outputs = sess_final.run(
    None,
    {
        "resnet_input": image.astype(np.float32),
        "pointnet_points": points.astype(np.float32)
    }
)
t1 = time.time()
print("合并后推理耗时: {:.2f} ms".format((t1-t0)*1000))

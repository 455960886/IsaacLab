# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    # def forward(self, image, state):
    #     image = self.normalizer(image)
    #     # 假设 actor 接受图像和状态拼接后的特征，或者分支处理
    #     return self.actor(image, state)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    # def forward(self, x):
    #     image_features = self.cnn_feature(x["image"])
    #     state_features = self.state_encoder(x["joint_pos"])
    #     # print("image_features : ",image_features.shape)
    #     # print("state_features : ",state_features.shape)
    #     observations = torch.cat((image_features, state_features), dim=1)
    #     return self.actor(observations)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            # obs = torch.zeros(1, self.actor[0].in_features)
            # torch.onnx.export(
            #     self,
            #     obs,
            #     os.path.join(path, filename),
            #     export_params=True,
            #     opset_version=11,
            #     verbose=self.verbose,
            #     input_names=["obs"],
            #     output_names=["actions"],
            #     dynamic_axes={},
            # )
        
        #改动
            dummy_input = torch.zeros(1, 1536)
            torch.onnx.export(
            self,
            dummy_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )

        #改动
        #     obs = {"image":torch.zeros(1, 3, 128, 128),"joint_pos":torch.zeros(6,)}
        # torch.onnx.export(
        #     self,
        #     obs,
        #     os.path.join(path, filename),
        #     export_params=True,
        #     opset_version=11,
        #     verbose=self.verbose,
        #     input_names=["obs"],
        #     output_names=["actions"],
        #     dynamic_axes={},
        # )
        import onnx
        
        import onnxruntime
        
        import cv2
        from PIL import Image

        onnx_model_path = '/home/roborock/IsaacLab/logs/rsl_rl/coarse_arm_lift/random_y_2/exported2/policy1905.onnx'
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_model_path)
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
            return
        else:
            print("The model is valid!")

        # Create inference session
        ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
        print("ONNX Runtime Providers:", ort_session.get_providers())

        # Load and preprocess image
        img = cv2.imread('/home/roborock/下载/test/test_5.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() 

        # print("pre img.shape:", img.shape)
        # print("pre img:", img)
      

        # Normalize image (zero mean)
        mean_tensor = torch.mean(img, dim=(0, 1), keepdim=True)
        img = (img-127.5)/255.0
        # Change shape to [N, C, H, W]
        img = img.permute(2, 0, 1).unsqueeze(0)
        # print("post img.shape:", img.shape)
        # print("post img:", img)

        # Convert to numpy
        img = img.cpu().numpy()

        # Inference
        ort_inputs = {"obs": img}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_output = ort_outputs[0]

        print("######### ort_output:", ort_output)
        return ort_output




# inference.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
# 引入量化所需的库
try:
    from bitsandbytes.nn import Linear4bit
    import bitsandbytes as bnb
    quantization_available = True
except ImportError:
    print("警告: bitsandbytes未安装，无法进行4-bit量化。请使用 'pip install bitsandbytes' 安装。")
    quantization_available = False

def chat_with_model(model_dir, question):
    """与微调后的模型对话"""
    # 清理GPU内存
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
    
    # 检查是否包含output前缀
    if model_dir.startswith("output/"):
        model_path = model_dir
    else:
        # 先检查新的模型目录结构
        new_model_path = os.path.join("output", "models", model_dir)
        if os.path.exists(new_model_path):
            model_path = new_model_path
        # 如果新路径不存在，再尝试旧的目录结构
        elif os.path.exists(os.path.join("output", model_dir)):
            model_path = os.path.join("output", model_dir)
        else:
            # 如果两者都不存在，返回错误
            return f"找不到模型: 已检查路径 'output/models/{model_dir}' 和 'output/{model_dir}' 但均不存在"
    
    print(f"找到微调模型路径: {model_path}")
    
    # 检测这是否是一个PEFT适配器模型
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)
    
    try:
        if is_peft:
            # 这是一个适配器模型 (P-Tuning或LoRA)
            print("检测到适配器模型")
            
            # 读取适配器配置获取基座模型路径
            with open(adapter_config_path, "r") as f:
                import json
                config = json.load(f)
                base_model_name = config.get('base_model_name_or_path')
                print(f"配置中的基座模型路径: {base_model_name}")
                
                # 解析基座模型路径
                # 常见格式: /root/models/Qwen/Qwen2.5-1.5B、Qwen2.5-1.5B等
                if os.path.exists(base_model_name):
                    # 如果是一个有效的绝对路径
                    base_model_path = base_model_name
                elif "Qwen" in base_model_name:
                    # 尝试从模型名构建路径
                    if "/" in base_model_name:
                        # 形如"Qwen/Qwen2.5-1.5B"的格式
                        model_name = base_model_name.split("/")[-1]
                    else:
                        # 形如"Qwen2.5-1.5B"的格式
                        model_name = base_model_name
                    
                    # 首先尝试常见的模型路径
                    potential_paths = [
                        os.path.expanduser(f"~/models/Qwen/{model_name}"),
                        os.path.expanduser(f"~/models/{model_name}"),
                        # 来自load_ptuning.py的增强路径查找
                        base_model_name,  # 原始路径
                        f"/root/models/Qwen/{model_name}",
                        f"/root/models/{model_name}",
                        # 移除路径中可能存在的双下划线(由训练脚本替换的斜杠)
                        os.path.expanduser(f"~/models/Qwen/{model_name.replace('__', '/')}"),
                        f"/root/models/Qwen/{model_name.replace('__', '/')}"
                    ]
                    
                    base_model_path = None
                    for path in potential_paths:
                        if os.path.exists(path):
                            base_model_path = path
                            print(f"找到基座模型: {base_model_path}")
                            break
                    
                    if not base_model_path:
                        return f"无法找到基座模型。请确保模型已下载到以下路径之一: {potential_paths}"
                else:
                    # 其他格式的模型名，尝试通用路径查找
                    model_name = os.path.basename(base_model_name)
                    potential_paths = [
                        base_model_name,  # 原始路径
                        os.path.expanduser(f"~/models/{model_name}"),
                        f"/root/models/{model_name}"
                    ]
                    
                    base_model_path = None
                    for path in potential_paths:
                        if os.path.exists(path):
                            base_model_path = path
                            print(f"找到基座模型: {base_model_path}")
                            break
                    
                    if not base_model_path:
                        return f"无法找到基座模型。请确保模型已下载到以下路径之一: {potential_paths}"
                
                # 修正适配器配置中的基座模型路径
                if base_model_path != base_model_name:
                    config['base_model_name_or_path'] = base_model_path
                    with open(adapter_config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    print(f"已更新基座模型路径为: {base_model_path}")
            
            # 使用正确的基座模型路径加载tokenizer和模型
            print(f"加载基座模型tokenizer: {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, 
                trust_remote_code=True,
                local_files_only=True
            )
            
            print(f"加载基座模型: {base_model_path}")
            # 根据是否安装了量化库决定使用哪种加载方式
            base_model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "device_map": "auto",
                "local_files_only": True,
                "low_cpu_mem_usage": True  # 减少CPU内存使用
            }
            
            # 如果安装了量化库，则添加量化参数
            if quantization_available:
                print("使用4bit量化加载模型以节省显存...")
                base_model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                })
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **base_model_kwargs
            )
            
            print(f"加载适配器: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path, local_files_only=True)
            print("模型加载完成")
        else:
            # 这是一个完整保存的模型
            print(f"加载完整模型: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "device_map": "auto",
                "local_files_only": True,
                "low_cpu_mem_usage": True  # 减少CPU内存使用
            }
            
            # 如果安装了量化库，则添加量化参数
            if quantization_available:
                print("使用4bit量化加载模型以节省显存...")
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                })
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                **model_kwargs
            )
        
        # 准备输入
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 格式化对话输入
        prompt = f"用户：{question}\n助手："
        
        print(f"开始生成回答...")
        
        # 执行推理，使用线程安全的超时机制
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 使用Threading模块实现超时，而不是signal
        import threading
        import time
        
        # 设置一个事件对象用于超时控制
        done_event = threading.Event()
        generation_result = {"response": None, "error": None}
        
        def generate_with_timeout():
            try:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=256,  # 减少生成的token数量
                    temperature=0.5,     # 降低温度，减少随机性
                    do_sample=True,
                    top_p=0.9,
                    top_k=40,           # 添加top_k参数限制选择范围
                    repetition_penalty=1.2,  # 增加重复惩罚
                    num_beams=1,        # 使用贪婪解码
                    max_time=60,        # 设置最大生成时间(秒)
                    early_stopping=True  # 启用早停
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取助手的回答
                try:
                    # 尝试只获取助手的回答部分
                    assistant_response = response.split("助手：")[-1].strip()
                    generation_result["response"] = assistant_response
                except:
                    # 如果提取失败，返回完整响应
                    generation_result["response"] = response
            except Exception as e:
                generation_result["error"] = str(e)
            finally:
                done_event.set()  # 标记生成已完成
        
        # 启动生成线程
        generation_thread = threading.Thread(target=generate_with_timeout)
        generation_thread.daemon = True
        generation_thread.start()
        
        # 等待生成完成或超时
        timeout_seconds = 120
        is_done = done_event.wait(timeout_seconds)
        
        if is_done:
            print("生成完成！")
            if generation_result["error"]:
                return f"生成过程中出错: {generation_result['error']}"
            return generation_result["response"]
        else:
            print("生成超时，返回部分结果")
            return "生成超时。模型处理时间过长，可能因为资源限制或模型较大。请尝试简化问题或使用较小的模型。"
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"模型加载或推理出错: {str(e)}\n\n详细错误:\n{error_trace}"

def handle_chat(model_name, question):
    """处理网页上的聊天请求，直接调用chat_with_model函数"""
    # 直接使用chat_with_model函数，不再通过subprocess调用load_ptuning.py
    return chat_with_model(model_name, question)


if __name__ == "__main__":
    # 测试代码，支持命令行调用
    import sys
    if len(sys.argv) < 2:
        print("用法: python inference.py <模型目录> [问题]")
        sys.exit(1)
        
    model_dir = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "你好，你是谁？"
    
    print(f"\n提问: {question}")
    response = chat_with_model(model_dir, question)
    print(f"\n回答: {response}")

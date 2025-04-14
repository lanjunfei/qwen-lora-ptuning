from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import subprocess
import threading
import os
import logging
import shutil
import humanize  # 需要安装：pip install humanize
from datetime import datetime

app = Flask(__name__)

# 模型存储路径
BASE_MODEL_PATH = os.path.expanduser("~/models/Qwen")
if not os.path.exists(BASE_MODEL_PATH):
    os.makedirs(BASE_MODEL_PATH, exist_ok=True)
OUTPUT_DIR = "output"
DOWNLOADING_MODELS = {}

# ---- 训练任务 ----
def run_training(mode, model_path, data_path, epochs):
    # 生成与训练输出相同的时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建分开的目录结构
    # 1. models/ - 存放模型文件
    # 2. logs/ - 存放训练日志
    base_name = f"{mode}_{model_path.replace('/', '_')}_{timestamp}"
    model_output_dir = f"output/models/{base_name}"
    logs_output_dir = f"output/logs/{base_name}"
    
    # 确保两个目录都存在
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)
    
    # 临时日志文件路径
    log_file_path = "training.log"
    
    # 清空当前日志文件
    open(log_file_path, "w").close()
    
    # 修改命令行参数，传递新的模型输出目录
    command = [
        "python3", "train.py",
        "--mode", mode,
        "--model_path", model_path,
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--output_dir", model_output_dir  # 添加输出目录参数
    ]
    
    log_file = open(log_file_path, "a", buffering=1)
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    # 添加一个线程在训练完成后将日志复制到日志目录
    def copy_log_when_done():
        process.wait()
        permanent_log_path = os.path.join(logs_output_dir, "training.log")
        shutil.copy2(log_file_path, permanent_log_path)
    
    thread = threading.Thread(target=copy_log_when_done)
    thread.daemon = True
    thread.start()
    
    return process

@app.route('/')
def index():
    model_dirs = []
    if os.path.exists(BASE_MODEL_PATH):
        for dir_name in os.listdir(BASE_MODEL_PATH):
            dir_path = os.path.join(BASE_MODEL_PATH, dir_name)
            if os.path.isdir(dir_path):
                model_dirs.append({'name': dir_name, 'path': dir_path})
    return render_template("index.html", models=model_dirs)

@app.route('/start_training', methods=["POST"])
def start_training():
    mode = request.form['mode']
    model_path = request.form['model_path']
    data_path = request.form['data_path']
    epochs = request.form['epochs']
    app.logger.debug(f"开始训练: mode={mode}, model_path={model_path}, data_path={data_path}, epochs={epochs}")
    open("training.log", "w").close()
    run_training(mode, model_path, data_path, epochs)
    return redirect("/log")

@app.route('/log')
def log():
    return render_template("log.html")

@app.route('/log_stream')
def log_stream():
    def generate():
        import time
        log_path = "training.log"
        while not os.path.exists(log_path):
            yield "data: 日志文件未创建，请等待...\n\n"
            time.sleep(1)
        with open(log_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                yield f"data:{line}\n\n"
    return Response(generate(), mimetype='text/event-stream')

# ---- 模型下载 ----
@app.route('/download_model', methods=["POST"])
def download_model():
    model_name = request.form['model_name']
    
    # 检查模型名称格式并处理
    if "/" in model_name:
        # 如果包含完整路径，如"Qwen/Qwen2.5-0.5B"
        repo_id = model_name
        base_name = model_name.split("/")[-1]  # 只取最后部分作为目录名
    else:
        # 如果只有模型名，如"Qwen2.5-0.5B"
        repo_id = "Qwen/" + model_name
        base_name = model_name  # 直接使用输入的名称
    
    # 创建保存路径 - 只使用模型的基本名称
    save_path = os.path.join(BASE_MODEL_PATH, base_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 创建下载日志文件
    log_path = os.path.join(save_path, "download.log")
    open(log_path, "w").close()  # 创建空文件
    
    def download():
        from huggingface_hub import snapshot_download
        import sys
        
        with open(log_path, "a", buffering=1) as log_file:
            try:
                log_file.write(f"开始下载模型: {repo_id}\n")
                log_file.write(f"保存路径: {save_path}\n")
                log_file.write("下载中...\n")
                log_file.flush()
                
                # 不使用任何进度回调
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=4,
                    token=os.getenv("HF_TOKEN")
                )
                log_file.write(f"模型 {repo_id} 下载完成！\n")
            except Exception as e:
                log_file.write(f"下载失败: {str(e)}\n")
                logging.error(f"下载失败: {str(e)}")
    
    thread = threading.Thread(target=download)
    thread.start()
    DOWNLOADING_MODELS[repo_id] = save_path
    return redirect(url_for('preload'))

@app.route('/download_log/<path:model_name>')
def download_log(model_name):
    path = DOWNLOADING_MODELS.get(model_name, "")
    log_path = os.path.join(path, "download.log")
    if not os.path.exists(log_path):
        return "日志不存在", 404
    def generate():
        with open(log_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    import time
                    time.sleep(0.5)
                    continue
                yield f"data:{line}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/preload')
def preload():
    return render_template("preload.html", models=DOWNLOADING_MODELS)

# ---- 推理管理 ----
@app.route('/inference')
def inference():
    models_list = []
    
    # 检查新结构目录
    new_models_dir = os.path.join(OUTPUT_DIR, "models")
    if os.path.exists(new_models_dir):
        for model in os.listdir(new_models_dir):
            if os.path.isdir(os.path.join(new_models_dir, model)):
                models_list.append(model)
    
    # 同时检查旧结构目录(向后兼容)
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path) and item != "models" and item != "logs" and not item.startswith('.'):
            models_list.append(item)
    
    return render_template("inference.html", models=models_list)

@app.route('/chat', methods=["POST"])
def chat():
    model_name = request.form['model']
    question = request.form['question']
    
    # 使用inference.py中的handle_chat函数
    from inference import handle_chat
    
    try:
        # 设置超时时间为5分钟
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def process_request():
            try:
                answer = handle_chat(model_name, question)
                result_queue.put({"success": True, "answer": answer})
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                result_queue.put({
                    "success": False, 
                    "answer": f"模型加载或推理出错: {str(e)}\n\n详细错误:\n{error_trace}"
                })
        
        # 启动处理线程
        thread = threading.Thread(target=process_request)
        thread.daemon = True
        thread.start()
        
        # 等待结果，最多等待5分钟
        try:
            result = result_queue.get(timeout=300)
            if result["success"]:
                return jsonify({"answer": result["answer"]})
            else:
                return jsonify({"answer": result["answer"]})
        except queue.Empty:
            return jsonify({"answer": "请求处理超时。大型模型可能需要较长处理时间，请尝试使用较小的模型或简化问题。"})
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return jsonify({"answer": f"处理请求时发生错误: {str(e)}\n\n详细错误:\n{error_trace}"})

@app.route('/api/models', methods=['GET'])
def list_models():
    """获取所有已下载的模型及其大小"""
    models_dir = "/root/models/Qwen"  # 根据实际路径调整
    models = []
    
    if os.path.exists(models_dir):
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                # 计算文件夹大小
                size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        size += os.path.getsize(fp)
                
                # 转换为易读格式
                readable_size = humanize.naturalsize(size)
                
                models.append({
                    "name": model_name,
                    "path": model_path,
                    "size": readable_size
                })
    
    return jsonify({"models": models})

@app.route('/api/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """删除指定模型"""
    models_dir = "/root/models/Qwen"  # 根据实际路径调整
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        return jsonify({"success": False, "message": "模型不存在"}), 404
    
    try:
        shutil.rmtree(model_path)
        return jsonify({"success": True, "message": f"已成功删除模型 {model_name}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500

# 新增：已微调模型的API
@app.route('/api/tuned_models', methods=['GET'])
def list_tuned_models():
    """获取所有已微调模型及其大小"""
    models_list = []
    
    # 检查新结构目录 - output/models/
    new_models_dir = os.path.join(OUTPUT_DIR, "models")
    if os.path.exists(new_models_dir):
        for model_name in os.listdir(new_models_dir):
            model_path = os.path.join(new_models_dir, model_name)
            if os.path.isdir(model_path):
                # 计算文件夹大小
                size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        size += os.path.getsize(fp)
                
                # 转换为易读格式
                readable_size = humanize.naturalsize(size)
                
                models_list.append({
                    "name": model_name,
                    "path": model_path,
                    "size": readable_size
                })
    
    # 同时检查旧结构目录 - output/
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path) and item != "models" and item != "logs" and not item.startswith('.'):
            # 计算文件夹大小
            size = 0
            for dirpath, dirnames, filenames in os.walk(item_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    size += os.path.getsize(fp)
            
            # 转换为易读格式
            readable_size = humanize.naturalsize(size)
            
            models_list.append({
                "name": item,
                "path": item_path,
                "size": readable_size
            })
    
    return jsonify({"models": models_list})

@app.route('/api/tuned_models/<model_name>', methods=['DELETE'])
def delete_tuned_model(model_name):
    """删除指定微调模型"""
    # 先检查新结构目录
    new_model_path = os.path.join(OUTPUT_DIR, "models", model_name)
    
    # 如果新结构目录不存在，再检查旧结构目录
    if not os.path.exists(new_model_path):
        new_model_path = os.path.join(OUTPUT_DIR, model_name)
        if not os.path.exists(new_model_path):
            return jsonify({"success": False, "message": "微调模型不存在"}), 404
    
    try:
        shutil.rmtree(new_model_path)
        return jsonify({"success": True, "message": f"已成功删除微调模型 {model_name}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9110, debug=True)
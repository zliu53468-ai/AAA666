import os
import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import tempfile
import gc

# Third-party libraries
import numpy as np
import joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS Configuration - Set origins to "*" for testing to allow all origins
# IMPORTANT: In a production environment, you should replace "*" with your specific frontend domain(s) for security reasons.
CORS(app, 
     origins="*", # Changed to allow all origins for debugging
     methods=["GET", "POST", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Reduced thread pool for lighter resource usage
executor = ThreadPoolExecutor(max_workers=1)

# Simplified Model Manager
class SimpleModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.ml_model = None
        self.ml_label_encoder = None
        self.markov_chain_model = defaultdict(lambda: defaultdict(int))
        self.is_initialized = False
        self.is_training = False
        self.last_training_time = 0
        
        # Use temp directory for model files
        self.temp_dir = tempfile.gettempdir()
        self.MODEL_FILE = os.path.join(self.temp_dir, 'baccarat_model.joblib')
        self.ENCODER_FILE = os.path.join(self.temp_dir, 'encoder.joblib')
    
    def set_training_status(self, status):
        """設定模型訓練狀態"""
        with self._lock:
            self.is_training = status
            if status:
                self.last_training_time = time.time()
    
    def get_training_status(self):
        """取得模型訓練狀態"""
        with self._lock:
            return self.is_training
    
    def should_skip_training(self, min_interval=60):
        """防止過於頻繁的訓練"""
        with self._lock:
            return (time.time() - self.last_training_time) < min_interval

# Initialize model manager
model_manager = SimpleModelManager()

# Reduced constants for lower resource usage
LOOK_BACK = 5
MARKOV_LOOK_BACK = 2
MIN_TRAINING_DATA = 20
MAX_HISTORY_SIZE = 3000  # Reduced for memory efficiency
LONG_RUN_THRESHOLD = 5 # 定義長龍的最小長度

# Simplified Game History
class GameHistory:
    def __init__(self, max_size=MAX_HISTORY_SIZE):
        self._history = []
        self._lock = threading.RLock()
        self._max_size = max_size
    
    def get_history(self):
        """取得歷史紀錄"""
        with self._lock:
            return self._history.copy()
    
    def set_history(self, new_history):
        """設定歷史紀錄，並截斷超過最大值的部份"""
        with self._lock:
            if len(new_history) > self._max_size:
                self._history = new_history[-self._max_size:]
                logger.warning(f"History truncated to {self._max_size} records")
            else:
                self._history = new_history.copy()
    
    def clear_history(self):
        """清除歷史紀錄"""
        with self._lock:
            self._history = []
    
    def size(self):
        """取得歷史紀錄大小"""
        with self._lock:
            return len(self._history)

game_history = GameHistory()

# Error handling decorator
def handle_errors(f):
    """錯誤處理裝飾器"""
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400
        except Exception as e:
            logger.error(f"Server error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500
    decorated_function.__name__ = f.__name__
    return decorated_function

# Simplified Markov chain functions
def update_markov_chain(history_data):
    """更新馬可夫鏈模型"""
    with model_manager._lock:
        model_manager.markov_chain_model = defaultdict(lambda: defaultdict(int))
        
        if len(history_data) < MARKOV_LOOK_BACK + 1:
            return
        
        for i in range(len(history_data) - MARKOV_LOOK_BACK):
            current_state = tuple(history_data[i : i + MARKOV_LOOK_BACK])
            next_state = history_data[i + MARKOV_LOOK_BACK]
            model_manager.markov_chain_model[current_state][next_state] += 1

def predict_markov_chain(history_slice):
    """馬可夫鏈預測"""
    if len(history_slice) < MARKOV_LOOK_BACK:
        return {'B': 1/3, 'P': 1/3, 'T': 1/3}
    
    current_state = tuple(history_slice[-MARKOV_LOOK_BACK:])
    
    with model_manager._lock:
        if current_state not in model_manager.markov_chain_model:
            return {'B': 1/3, 'P': 1/3, 'T': 1/3}
        
        transitions = model_manager.markov_chain_model[current_state]
        total_transitions = sum(transitions.values())
        
        if total_transitions == 0:
            return {'B': 1/3, 'P': 1/3, 'T': 1/3}
        
        probabilities = {
            outcome: count / total_transitions
            for outcome, count in transitions.items()
        }
        
        for outcome in ['B', 'P', 'T']:
            if outcome not in probabilities:
                probabilities[outcome] = 0.0
        
        return probabilities

# 新增：獲取牌路趨勢數據
def get_run_trends(history_data):
    """
    分析歷史數據以獲取連續牌路（龍）和斷點資訊。
    返回 {
        'current_run_type': 'B' or 'P' or None,
        'current_run_length': int,
        'last_breakpoint_type': 'B' or 'P' or None, # 被斷的龍的類型
        'last_breakpoint_length': int, # 被斷的龍的長度
        'is_long_run_breakpoint': bool # 是否為長龍斷點
    }
    """
    trends = {
        'current_run_type': None,
        'current_run_length': 0,
        'last_breakpoint_type': None,
        'last_breakpoint_length': 0,
        'is_long_run_breakpoint': False
    }

    if not history_data:
        return trends

    current_run_type = history_data[-1]
    current_run_length = 0
    
    # 計算當前連續次數
    for i in range(len(history_data) - 1, -1, -1):
        if history_data[i] == current_run_type:
            current_run_length += 1
        else:
            break
    
    trends['current_run_type'] = current_run_type
    trends['current_run_length'] = current_run_length

    # 尋找最近的斷點
    if len(history_data) > current_run_length:
        prev_result_index = len(history_data) - current_run_length - 1
        prev_run_type = history_data[prev_result_index]
        prev_run_length = 0
        for i in range(prev_result_index, -1, -1):
            if history_data[i] == prev_run_type:
                prev_run_length += 1
            else:
                break
        
        trends['last_breakpoint_type'] = prev_run_type
        trends['last_breakpoint_length'] = prev_run_length
        
        # 判斷是否為長龍斷點
        if prev_run_length >= LONG_RUN_THRESHOLD:
            trends['is_long_run_breakpoint'] = True

    return trends

# Simplified feature engineering
def encode_result_for_ml(result):
    """將結果編碼為機器學習可用數值"""
    if result == 'B':
        return 0
    elif result == 'P':
        return 1
    elif result == 'T':
        return 2
    return 0

def get_simple_features(history_data):
    """提取機器學習的簡單特徵，加入牌路趨勢"""
    if len(history_data) < LOOK_BACK:
        return None
    
    features = []
    
    # 最近的結果作為數值特徵
    recent = history_data[-LOOK_BACK:]
    for result in recent:
        features.append(encode_result_for_ml(result))
    
    # 基本比例
    total_len = len(history_data)
    features.append(history_data.count('B') / total_len)
    features.append(history_data.count('P') / total_len)
    features.append(history_data.count('T') / total_len)
    
    # 連續次數 (原始邏輯)
    if history_data:
        current = history_data[-1]
        streak = 1
        # 修正: range 應該從 1 開始，上限應為 min(len(history_data), 10)
        for i in range(1, min(len(history_data), 10)):  # 最多連續10次
            if history_data[-i-1] == current: # 修正索引
                streak += 1
            else:
                break
        features.append(streak)
    else:
        features.append(0)
    
    # 新增：牌路趨勢特徵
    run_trends = get_run_trends(history_data)
    features.append(run_trends['current_run_length'])
    features.append(1 if run_trends['current_run_type'] == 'B' else (0 if run_trends['current_run_type'] == 'P' else -1)) # 莊:1, 閒:0, 無:-1
    features.append(run_trends['last_breakpoint_length'])
    features.append(1 if run_trends['last_breakpoint_type'] == 'B' else (0 if run_trends['last_breakpoint_type'] == 'P' else -1)) # 莊:1, 閒:0, 無:-1
    features.append(1 if run_trends['is_long_run_breakpoint'] else 0)

    return np.array(features)

# Simplified model training
def prepare_training_data(history_data):
    """準備訓練數據"""
    X_features = []
    y = []
    
    if len(history_data) < MIN_TRAINING_DATA:
        return np.array([]), np.array([])
    
    # 調整訓練數據的生成方式，確保特徵與標籤對齊
    # 這裡我們需要足夠的歷史數據來生成 LOOK_BACK + 趨勢特徵
    min_context_for_features = LOOK_BACK + 1 # 至少需要 LOOK_BACK + 1 才能生成完整的趨勢特徵

    for i in range(min_context_for_features, len(history_data)):
        context = history_data[:i] # 使用當前點之前的所有歷史數據來生成特徵
        features = get_simple_features(context)
        
        if features is not None and len(features) == 13: # 確保特徵數量正確 (LOOK_BACK(5) + 3個比例 + 1個連勝 + 4個趨勢 = 13)
            X_features.append(features)
            y.append(encode_result_for_ml(history_data[i])) # 預測 context 後的結果
        else:
            logger.warning(f"Skipping training data point at index {i} due to incomplete features. Feature length: {len(features) if features is not None else 'None'}")
    
    return np.array(X_features), np.array(y)

def train_simple_model(history_data):
    """訓練簡單模型"""
    X_features, y_labels = prepare_training_data(history_data)
    
    if len(X_features) == 0:
        logger.warning("No sufficient training data available for ML model.")
        return False, "No sufficient training data available"
    
    try:
        with model_manager._lock:
            # 使用簡單的 RandomForest 以獲得更好的兼容性
            model_manager.ml_model = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
                random_state=42,
                n_jobs=1  # 單線程以提高穩定性
            )
            
            model_manager.ml_model.fit(X_features, y_labels)
            
            # 初始化標籤編碼器
            if model_manager.ml_label_encoder is None:
                model_manager.ml_label_encoder = LabelEncoder()
                # 確保編碼器能處理所有可能的結果
                model_manager.ml_label_encoder.fit(['B', 'P', 'T'])
            
            # 將模型儲存到臨時目錄
            try:
                joblib.dump(model_manager.ml_model, model_manager.MODEL_FILE)
                joblib.dump(model_manager.ml_label_encoder, model_manager.ENCODER_FILE)
                logger.info("Model saved successfully")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
            
            model_manager.is_initialized = True
            
            # 強制垃圾回收
            gc.collect()
            
            return True, "Model trained successfully"
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False, f"Training failed: {str(e)}"

def load_simple_models():
    """載入現有模型"""
    try:
        if os.path.exists(model_manager.MODEL_FILE):
            model_manager.ml_model = joblib.load(model_manager.MODEL_FILE)
            logger.info("Model loaded successfully")
        
        if os.path.exists(model_manager.ENCODER_FILE):
            model_manager.ml_label_encoder = joblib.load(model_manager.ENCODER_FILE)
        else:
            model_manager.ml_label_encoder = LabelEncoder()
            model_manager.ml_label_encoder.fit(['B', 'P', 'T'])
        
        model_manager.is_initialized = model_manager.ml_model is not None
        return model_manager.is_initialized
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        model_manager.ml_label_encoder = LabelEncoder()
        model_manager.ml_label_encoder.fit(['B', 'P', 'T'])
        return False

def predict_next_outcome(history_data):
    """預測下一個結果，加入長龍斷點的策略性觀望"""
    # 如果模型尚未初始化，則嘗試載入模型
    if not model_manager.is_initialized:
        load_simple_models()

    if not history_data:
        return {
            "prediction": "OBSERVE",
            "probabilities": {'B': 1/3, 'P': 1/3, 'T': 1/3},
            "confidence": 1/3,
            "source": "no_data"
        }
    
    results = {}
    
    # 馬可夫鏈預測
    if len(history_data) >= MARKOV_LOOK_BACK:
        markov_probs = predict_markov_chain(history_data)
        markov_pred = max(markov_probs, key=markov_probs.get)
        results['markov'] = {
            "prediction": markov_pred,
            "probabilities": markov_probs,
            "confidence": markov_probs[markov_pred],
            "source": "markov"
        }
    
    # 機器學習預測
    if (model_manager.ml_model is not None and 
        len(history_data) >= MIN_TRAINING_DATA):
        try:
            features = get_simple_features(history_data)
            if features is not None and len(features) == 13: # 再次檢查特徵數量
                with model_manager._lock:
                    probas = model_manager.ml_model.predict_proba([features])[0]
                    pred_idx = np.argmax(probas)
                    
                    # 轉換為結果標籤
                    # 確保所有可能的標籤都在編碼器中
                    outcome_map = {idx: label for label, idx in zip(model_manager.ml_label_encoder.classes_, model_manager.ml_label_encoder.transform(model_manager.ml_label_encoder.classes_))}
                    ml_pred = outcome_map.get(pred_idx, 'OBSERVE') # 如果索引不在map中，預設為觀察
                    
                    ml_prob_dict = {
                        outcome_map.get(i, 'UNKNOWN'): float(probas[i]) if i < len(probas) else 0.0
                        for i in range(len(probas))
                    }
                    
                    results['ml'] = {
                        "prediction": ml_pred,
                        "probabilities": ml_prob_dict,
                        "confidence": float(probas[pred_idx]),
                        "source": "ml"
                    }
            else:
                logger.warning(f"ML prediction skipped due to incomplete features. Feature length: {len(features) if features is not None else 'None'}")
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
    
    # ----------------------------------------------------
    # 新增：長龍斷點的策略性觀望邏輯 (調整)
    # ----------------------------------------------------
    run_trends = get_run_trends(history_data)
    if run_trends['is_long_run_breakpoint'] and run_trends['current_run_length'] == 1:
        logger.info(f"Detected long run breakpoint for {run_trends['last_breakpoint_type']} (length {run_trends['last_breakpoint_length']}). Suggesting OBSERVE.")
        return {
            "prediction": "OBSERVE",
            "probabilities": {'B': 1/3, 'P': 1/3, 'T': 1/3}, # 在觀望時給予平均機率
            "confidence": 0.35, # 稍微降低信心度，表示不確定
            "source": "strategic_observe_breakpoint",
            "trend_info": run_trends # 將趨勢資訊一同返回
        }

    # 選擇最佳結果 (如果沒有策略性觀望)
    if 'ml' in results and results['ml']['confidence'] > 0.4: # 降低信心度閾值，更容易建議下注
        return results['ml']
    elif 'markov' in results and results['markov']['confidence'] > 0.4: # 降低信心度閾值
        return results['markov']
    else:
        # 如果都沒有達到高信心度，仍建議觀望，但給出理由
        return {
            "prediction": "OBSERVE",
            "probabilities": {'B': 1/3, 'P': 1/3, 'T': 1/3},
            "confidence": 1/3,
            "source": "low_confidence_default_observe" # 增加觀望原因
        }

# Input validation
def validate_history_data(data):
    """驗證歷史數據 (接受字典或列表)"""
    validated_history = []

    if isinstance(data, dict):
        history = data.get('history', [])
        if not isinstance(history, list):
            raise ValueError("History must be an array within the JSON object.")
    elif isinstance(data, list):
        history = data
    else:
        raise ValueError("Input data must be a JSON object with 'history' key or a JSON array.")
    
    valid_outcomes = {'B', 'P', 'T'}
    
    for item in history:
        if not isinstance(item, str):
            raise ValueError(f"Invalid outcome type: {type(item)}. Expected string 'B', 'P', or 'T'.")
        
        item_upper = item.upper()
        if item_upper not in valid_outcomes:
            raise ValueError(f"Invalid outcome: '{item}'. Expected 'B', 'P', or 'T'.")
        
        validated_history.append(item_upper)
    
    return validated_history

# Flask routes
@app.route('/')
@handle_errors
def home():
    """API 首頁"""
    return jsonify({
        "message": "Baccarat Prediction API (Koyeb Optimized)",
        "status": "running",
        "version": "2.1 (with trend analysis)"
    })

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    """預測下一個結果"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400
    
    history = validate_history_data(data) # 使用更新後的驗證
    result = predict_next_outcome(history)
    
    return jsonify(result)

@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
@handle_errors
def handle_history_api():
    """處理歷史紀錄 API"""
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({"error": "Please provide JSON data"}), 400
        
        # 使用更新後的 validate_history_data 函數
        try:
            history_data = validate_history_data(data)
        except ValueError as e:
            logger.warning(f"History API validation error: {e}")
            return jsonify({"error": f"Invalid history data: {str(e)}"}), 400
        
        game_history.set_history(history_data)
        
        # 異步訓練，降低頻率
        if (len(history_data) >= MIN_TRAINING_DATA and 
            not model_manager.should_skip_training() and
            not model_manager.get_training_status()):
            
            def async_train():
                try:
                    model_manager.set_training_status(True)
                    success, message = train_simple_model(history_data)
                    logger.info(f"Async training: {message}")
                except Exception as e:
                    logger.error(f"Async training failed: {e}")
                finally:
                    model_manager.set_training_status(False)
            
            executor.submit(async_train)
        
        # 總是更新馬可夫鏈
        update_markov_chain(history_data)
        
        return jsonify({
            "message": "History saved successfully",
            "records": len(history_data)
        }), 200
        
    elif request.method == 'GET':
        return jsonify(game_history.get_history()), 200
        
    elif request.method == 'DELETE':
        game_history.clear_history()
        model_manager.is_initialized = False # 清除歷史後重置模型狀態
        return jsonify({"message": "History cleared successfully"}), 200

@app.route('/status', methods=['GET'])
@handle_errors
def status():
    """檢查模型狀態"""
    return jsonify({
        "model_loaded": model_manager.ml_model is not None,
        "encoder_loaded": model_manager.ml_label_encoder is not None,
        "current_history_length": game_history.size(),
        "is_initialized": model_manager.is_initialized,
        "is_training": model_manager.get_training_status(),
        "memory_optimized": True,
        "version": "2.1 (with trend analysis)"
    })

@app.route('/train', methods=['POST'])
@handle_errors
def train_model_endpoint():
    """模型訓練端點"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400
    
    history = validate_history_data(data)
    success, message = train_simple_model(history)
    
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 400

@app.route('/recommendation', methods=['POST'])
@handle_errors
def get_recommendation():
    """取得下注建議，加入長龍斷點的反向下注考量"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400
    
    history = validate_history_data(data)
    result = predict_next_outcome(history)
    
    prediction = result["prediction"]
    confidence = result["confidence"]
    source_model = result["source"]
    
    recommendation_text = ""
    bet_amount_text = ""

    # 調整信心度閾值，讓 AI 更常給出下注建議
    # 新的信心度閾值為 0.40
    CONFIDENCE_THRESHOLD_LIGHT_BET = 0.40
    CONFIDENCE_THRESHOLD_OBSERVE_LOW_CONFIDENCE = 0.30 # 更低的閾值，用來區分極低信心度

    if source_model == "strategic_observe_breakpoint":
        trend_info = result.get("trend_info", {})
        last_breakpoint_type = trend_info.get('last_breakpoint_type')
        reverse_bet_direction_chinese = None
        reverse_bet_direction_english = None

        if last_breakpoint_type == 'B':
            reverse_bet_direction_chinese = '閒'
            reverse_bet_direction_english = 'P'
        elif last_breakpoint_type == 'P':
            reverse_bet_direction_chinese = '莊'
            reverse_bet_direction_english = 'B'

        if reverse_bet_direction_chinese:
            recommendation_text = f"長龍斷點！前長龍為 {last_breakpoint_type}，建議觀望，可考慮反押 {reverse_bet_direction_chinese} ({reverse_bet_direction_english})。"
            bet_amount_text = "Small (Cautious)" # 建議非常小的下注
        else:
            recommendation_text = "長龍斷點！趨勢不明，建議觀望。"
            bet_amount_text = "No bet" # 依然保守
    elif prediction == "OBSERVE": # 如果 AI 預測就是 OBSERVE
        if source_model == "low_confidence_default_observe":
            recommendation_text = "AI 信心度不足，建議觀望。"
        else:
            recommendation_text = "目前趨勢不明顯，建議觀望。"
        bet_amount_text = "No bet"
    elif confidence < CONFIDENCE_THRESHOLD_OBSERVE_LOW_CONFIDENCE: # 極低信心度
        recommendation_text = "AI 信心度極低，建議觀望。"
        bet_amount_text = "No bet"
    elif confidence < CONFIDENCE_THRESHOLD_LIGHT_BET: # 略低於「輕微下注」閾值，但高於極低信心
        recommendation_text = f"AI 信心度一般，可考慮輕微下注 {prediction}。"
        bet_amount_text = "Very Small"
    else: # 信心度足夠進行輕微下注
        recommendation_text = f"AI 建議下注 {prediction}。"
        bet_amount_text = "Small"
    
    return jsonify({
        "prediction": prediction,
        "probabilities": result["probabilities"],
        "confidence": confidence,
        "recommendation": recommendation_text, # 使用新的建議文字
        "bet_amount": bet_amount_text,       # 使用新的下注金額文字
        "source_model": source_model
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Koyeb 健康檢查"""
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    """內部伺服器錯誤處理"""
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """資源未找到錯誤處理"""
    return jsonify({"error": "Resource not found"}), 404

# Cleanup
import atexit
def cleanup():
    """程式退出時清理執行緒池"""
    executor.shutdown(wait=False)
atexit.register(cleanup)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # 在生產環境中始終禁用調試模式
        threaded=True
    )

from flask import Flask, request, render_template, jsonify
from rationality_analyzer import RationalityAnalyzer
from contradiction_detector import AdvancedContradictionDetector
import time
import threading
import traceback
import sys

# Создаем приложение Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Глобальные переменные для ленивой загрузки
rationality_analyzer = None
contradiction_detector = None
models_loaded = False
models_loading = False

def load_models():
    """Загружает модели в фоновом потоке"""
    global rationality_analyzer, contradiction_detector, models_loaded, models_loading
    
    try:
        models_loading = True
        print("🔄 Loading Rationality Analyzer...")
        rationality_analyzer = RationalityAnalyzer()
        print("✅ Rationality Analyzer loaded")
        
        print("🔄 Loading Contradiction Detector (this may take 1-2 minutes)...")
        contradiction_detector = AdvancedContradictionDetector()
        print("✅ Contradiction Detector loaded")
        
        models_loaded = True
        models_loading = False
        print("🎉 All models ready!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        models_loading = False

# Запускаем загрузку моделей в фоновом потоке
threading.Thread(target=load_models, daemon=True).start()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/status')
def status():
    """Проверяет статус загрузки моделей"""
    return jsonify({
        'models_loaded': models_loaded,
        'models_loading': models_loading
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Анализирует текст на рациональность и логические противоречия.
    
    Ожидает JSON: {"text": "текст для анализа"}
    Возвращает JSON с результатами анализа
    """
    global models_loaded, models_loading
    
    # Проверяем статус загрузки моделей
    if models_loading:
        return jsonify({
            'error': 'Models are still loading. Please wait 1-2 minutes and try again.',
            'status': 'loading'
        }), 202
    
    if not models_loaded:
        return jsonify({
            'error': 'Models failed to load. Please restart the server.',
            'status': 'error'
        }), 500
    
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        text = data.get('text', '').strip()
        
        # Проверяем минимальную длину текста
        if not text or len(text) < 10:
            return jsonify({
                'error': 'Text is too short (minimum 10 characters required)'
            }), 400
        
        start_time = time.time()
        
        # Анализируем рациональность
        try:
            rationality_result = rationality_analyzer.analyze(text)
        except Exception as e:
            return jsonify({
                'error': f'Rationality analysis failed: {str(e)}',
                'details': traceback.format_exc()
            }), 500
        
        # Анализируем противоречия
        try:
            contradictions = contradiction_detector.detect_contradictions(text)
            logic_result = contradiction_detector.generate_report(contradictions, len(text))
        except Exception as e:
            return jsonify({
                'error': f'Contradiction analysis failed: {str(e)}',
                'details': traceback.format_exc()
            }), 500
        
        # Вычисляем итоговую оценку
        overall_score = (rationality_result['rationality_score'] + logic_result['consistency_score']) / 2
        
        # Генерируем общую рекомендацию
        overall_recommendation = generate_overall_recommendation(overall_score)
        
        processing_time = round(time.time() - start_time, 2)
        
        # Подготавливаем детальный разбор
        # Извлекаем числовые значения для передачи на фронтенд
        emotional_words = rationality_result['emotional_markers']['count']
        logical_words = rationality_result['logical_connectors']['count']
        scientific_words = rationality_result['scientific_terms']['count']
        
        # Формируем результат
        result = {
            'rationality': {
                'score': float(rationality_result['rationality_score']),
                'analysis': str(rationality_result['analysis']),
                'emotional_markers': float(emotional_words),
                'logical_connectors': float(logical_words),
                'scientific_terms': float(scientific_words),
                'breakdown': rationality_result.get('breakdown', {})
            },
            'logic': {
                'score': float(logic_result['consistency_score']),
                'summary': str(logic_result['summary']),
                'total_contradictions': int(logic_result['total']),
                'by_type': logic_result.get('by_type', {}),
                'recommendation': str(logic_result['recommendation']),
                'contradictions': logic_result.get('contradictions', []),
                'quality_metrics': logic_result.get('quality_metrics', {})
            },
            'overall': {
                'score': round(float(overall_score), 1),
                'recommendation': str(overall_recommendation),
                'processing_time': float(processing_time),
                'quality_level': get_quality_level(overall_score)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        # Ловим все неожиданные ошибки
        error_msg = f'Analysis failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'error': error_msg,
            'details': traceback.format_exc()
        }), 500

def generate_overall_recommendation(score: float) -> str:
    """
    Генерирует общую рекомендацию на основе итоговой оценки.
    
    Аргументы:
        score (float): Итоговая оценка (0-100)
        
    Возвращает:
        str: Рекомендация
    """
    if score >= 90:
        return "Excellent! Text is highly rational and logically consistent."
    elif score >= 80:
        return "Very good. Text demonstrates strong rationality and logic."
    elif score >= 70:
        return "Good. Minor improvements could enhance clarity and logic."
    elif score >= 60:
        return "Acceptable. Some emotional influence and logical gaps present."
    elif score >= 50:
        return "Moderate. Significant improvements recommended for rationality and consistency."
    elif score >= 40:
        return "Poor. Text contains substantial emotional bias and logical errors."
    elif score >= 30:
        return "Very poor. High emotional content and serious logical contradictions."
    else:
        return "Critical. Text is dominated by emotions with no logical coherence."

def get_quality_level(score: float) -> str:
    """
    Преобразует оценку в уровень качества.
    
    Аргументы:
        score (float): Оценка (0-100)
        
    Возвращает:
        str: Уровень качества
    """
    if score >= 90:
        return 'excellent'
    elif score >= 80:
        return 'very_good'
    elif score >= 70:
        return 'good'
    elif score >= 60:
        return 'acceptable'
    elif score >= 50:
        return 'moderate'
    elif score >= 40:
        return 'poor'
    elif score >= 30:
        return 'very_poor'
    else:
        return 'critical'

@app.route('/api')
def api_info():
    """Документация API"""
    return jsonify({
        'name': 'Advanced Logic Contradiction Detector API',
        'version': '2.0.0',
        'description': 'API for analyzing text rationality and logical consistency',
        'endpoints': {
            'POST /analyze': {
                'description': 'Analyze text for rationality and logical contradictions',
                'request': {
                    'body': {
                        'text': 'string (minimum 10 characters)'
                    }
                },
                'response': {
                    'rationality': {
                        'score': 'float (0-100)',
                        'analysis': 'string',
                        'emotional_markers': 'float',
                        'logical_connectors': 'float',
                        'scientific_terms': 'float',
                        'breakdown': 'object'
                    },
                    'logic': {
                        'score': 'float (0-100)',
                        'summary': 'string',
                        'total_contradictions': 'int',
                        'by_type': 'object',
                        'recommendation': 'string',
                        'contradictions': 'array',
                        'quality_metrics': 'object'
                    },
                    'overall': {
                        'score': 'float (0-100)',
                        'recommendation': 'string',
                        'processing_time': 'float (seconds)',
                        'quality_level': 'string'
                    }
                }
            },
            'GET /status': {
                'description': 'Check if models are loaded',
                'response': {
                    'models_loaded': 'boolean',
                    'models_loading': 'boolean'
                }
            }
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Обработчик ошибки 404"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработчик ошибки 500"""
    return jsonify({
        'error': 'Internal server error',
        'status': 500,
        'details': str(error)
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Advanced Logic Contradiction Detector")
    print("=" * 60)
    print("🌐 Server will be available at http://localhost:5000")
    print("⏳ Models are loading in background...")
    print("💡 First analysis may take 1-2 minutes (model loading)")
    print("=" * 60)
    
    # Запускаем сервер
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)